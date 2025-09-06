import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List
from dataclasses import dataclass

LM_PORTS = {3389, 445, 5985, 5986, 135, 139, 22}  # RDP, SMB, WinRM, WMI/DCOM, SSH

@dataclass
class LMEdgeData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor]
    y_edge: torch.Tensor
    train_mask_edge: Optional[torch.Tensor] = None
    val_mask_edge: Optional[torch.Tensor] = None
    test_mask_edge: Optional[torch.Tensor] = None

def _minmax_norm(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a.astype(np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi == lo:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)

def build_from_ip_port_flow_csv(
    csv_path: str,
    src_ip_col: str = "Source IP",
    dst_ip_col: str = "Destination IP",
    src_port_col: str = "Source Port",          # optional if present
    dst_port_col: str = "Destination Port",
    label_col: str = "Label",
    time_col: Optional[str] = None,             # "Timestamp" if present
    chunksize: int = 200_000,                   # stream to avoid OOM / slow parses
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> LMEdgeData:
    """
    Nodes: unique IPs (hosts). Edge per flow row: SourceIP -> DestinationIP.
    Edge features: destination port + (if present) SYN/ACK/etc + coarse timing.
    Labels (edge): LM if (attack & LM-port) OR (benign & LM-port & handshake).
    """
    # Columns we will try to read (only load what we need)
    base_cols = [src_ip_col, dst_ip_col, dst_port_col, label_col]
    maybe_cols = [
        src_port_col, "SYN Flag Count", "ACK Flag Count",
        "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    ]
    usecols = list(dict.fromkeys(base_cols + ([time_col] if time_col else []) + maybe_cols))

    # storage
    node_index: Dict[str, int] = {}
    src_idx_list: List[int] = []
    dst_idx_list: List[int] = []
    edge_feat_list: List[np.ndarray] = []
    y_edge_list: List[int] = []

    # degree + hour buckets (for node features)
    deg_in: Dict[int, int] = {}
    deg_out: Dict[int, int] = {}
    hour_hist: Dict[int, np.ndarray] = {}

    def _nid(ip: str) -> int:
        if ip not in node_index:
            node_index[ip] = len(node_index)
        return node_index[ip]

    # stream the csv
    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c in usecols,
        chunksize=chunksize,
        low_memory=True,
        dtype=str,    # read as str then coerce selectively; faster & robust
        engine="c",
    )

    row_count = 0
    for chunk in reader:
        # strip/clean columns on the fly
        chunk.columns = [" ".join(c.strip().split()) for c in chunk.columns]

        # coerce numerics
        for col in [dst_port_col, src_port_col, "SYN Flag Count", "ACK Flag Count",
                    "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
                    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # time -> hour
        if time_col and time_col in chunk.columns:
            t = pd.to_datetime(chunk[time_col], errors="coerce")
            hour = t.dt.hour.fillna(0).astype(int).to_numpy()
        else:
            hour = np.zeros(len(chunk), dtype=int)

        # minimal features we always try to add
        dst_port = chunk[dst_port_col].fillna(0).astype("float32").to_numpy()
        syn = (chunk["SYN Flag Count"].fillna(0).astype("float32").to_numpy()
               if "SYN Flag Count" in chunk.columns else np.zeros(len(chunk), dtype=np.float32))
        ack = (chunk["ACK Flag Count"].fillna(0).astype("float32").to_numpy()
               if "ACK Flag Count" in chunk.columns else np.zeros(len(chunk), dtype=np.float32))

        # optional dynamics (if present)
        def getf(col, default=0.0):
            if col in chunk.columns:
                return chunk[col].fillna(0).astype("float32").to_numpy()
            return np.full(len(chunk), default, dtype=np.float32)

        flow_dur   = getf("Flow Duration")
        flow_ps    = getf("Flow Packets/s")
        flow_bs    = getf("Flow Bytes/s")
        iat_mean   = getf("Flow IAT Mean")
        iat_std    = getf("Flow IAT Std")
        iat_max    = getf("Flow IAT Max")
        iat_min    = getf("Flow IAT Min")

        # labels
        lbl = chunk[label_col].astype(str).str.upper().str.strip().to_numpy()
        is_attack = (lbl != "BENIGN").astype(np.int8)
        is_lm_port = np.isin(dst_port.astype(np.int64), list(LM_PORTS)).astype(np.int8)
        handshake = ((syn > 0) & (ack > 0)).astype(np.int8)
        y_edge = ((is_attack & is_lm_port) | ((1 - is_attack) & is_lm_port & handshake)).astype(np.int8)

        sips = chunk[src_ip_col].astype(str).fillna("0.0.0.0").to_numpy()
        dips = chunk[dst_ip_col].astype(str).fillna("0.0.0.0").to_numpy()

        # build per-row items
        for i in range(len(chunk)):
            s = _nid(sips[i]); d = _nid(dips[i])
            src_idx_list.append(s); dst_idx_list.append(d)
            y_edge_list.append(int(y_edge[i]))

            # edge features: [dst_port, syn, ack, hour, flow_dur, flow_ps, flow_bs, iat_mean, iat_std, iat_max, iat_min]
            edge_feat_list.append(np.array([
                dst_port[i], syn[i], ack[i], float(hour[i]),
                flow_dur[i], flow_ps[i], flow_bs[i],
                iat_mean[i], iat_std[i], iat_max[i], iat_min[i]
            ], dtype=np.float32))

            # degrees
            deg_out[s] = deg_out.get(s, 0) + 1
            deg_in[d]  = deg_in.get(d, 0) + 1
            # hour histogram for destination
            if d not in hour_hist:
                hour_hist[d] = np.zeros(24, dtype=np.float32)
            hour_hist[d][hour[i] % 24] += 1

        row_count += len(chunk)

    # tensors
    E = len(src_idx_list)
    N = len(node_index)
    edge_index = torch.tensor(
        np.vstack([np.asarray(src_idx_list, dtype=np.int64),
                   np.asarray(dst_idx_list, dtype=np.int64)]),
        dtype=torch.long
    )
    edge_attr = torch.tensor(np.vstack(edge_feat_list), dtype=torch.float32)
    y_edge = torch.tensor(np.asarray(y_edge_list, dtype=np.int64), dtype=torch.long)

    # node features: in/out degree (minmax), + 8 bucketed hours for dst host
    x = np.zeros((N, 16), dtype=np.float32)
    indeg = np.zeros(N, dtype=np.float32); outdeg = np.zeros(N, dtype=np.float32)
    for n in range(N):
        indeg[n] = float(deg_in.get(n, 0)); outdeg[n] = float(deg_out.get(n, 0))
    x[:, 0] = _minmax_norm(indeg)
    x[:, 1] = _minmax_norm(outdeg)
    for n in range(N):
        h = hour_hist.get(n, np.zeros(24, dtype=np.float32))
        h = _minmax_norm(h)
        buckets = np.array([h[i:i+3].mean() for i in range(0, 24, 3)], dtype=np.float32)  # 8 buckets
        x[n, 2:10] = buckets
    x = torch.tensor(x, dtype=torch.float32)

    # time-based split masks (by input order)
    e_train = int(E * train_ratio)
    e_val   = int(E * (train_ratio + val_ratio))
    train_mask = torch.zeros(E, dtype=torch.bool); train_mask[:e_train] = True
    val_mask   = torch.zeros(E, dtype=torch.bool); val_mask[e_train:e_val] = True
    test_mask  = torch.zeros(E, dtype=torch.bool); test_mask[e_val:] = True

    return LMEdgeData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_edge=y_edge,
        train_mask_edge=train_mask,
        val_mask_edge=val_mask,
        test_mask_edge=test_mask,
    )
