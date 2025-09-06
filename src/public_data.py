import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_headers(cols):
    return [" ".join(str(c).strip().split()) for c in cols]

def _normalize_label(s: str) -> str:
    s = str(s).replace("–", "-").replace("—", "-")
    return " ".join(s.strip().split()).upper()

# IMPORTANT: order must cover your action_node_idx count (Na)
_ATTACK_TYPES_CANON = [
    "WEB ATTACK - BRUTE FORCE",
    "DOS SLOWLORIS",
    "FTP-PATATOR",
    "SSH-PATATOR",
    "DDOS",
    "BOT",
    "PORTSCAN",
]

# ──────────────────────────────────────────────────────────────────────────────
# Main loader (streams CSV; fixes width/reshape safely)  — used by dataset2
# ──────────────────────────────────────────────────────────────────────────────
def load_CICIDS(
    num_benign: int,
    num_malic: int,
    action_node_idx,
    csv_path: str = "../datasets/public/CICIDS-2017.csv",
    chunksize: int = 200_000,
    random_seed: int = 1234,
    Fp_target: int = 11,          # <<< per-node feature size you want (fixes 7*11 case)
):
    """
    Returns:
      x_benign: [num_benign, Na, Fp_target]
      y_benign: [num_benign, Na]
      x_malic:  [num_malic*Na, Na, Fp_target]
      y_malic:  [num_malic*Na, Na]
    """
    rng = np.random.default_rng(random_seed)
    Na = len(action_node_idx)
    target_F = Na * Fp_target

    # How many benign rows needed to build benign set + base for malic set
    total_benign_needed = num_benign * Na + num_malic * Na * Na

    Xb_parts = []                                 # benign, unscaled
    attack_to_parts = {atk: [] for atk in _ATTACK_TYPES_CANON}  # per attack

    need_b = total_benign_needed
    need_attacks = {atk: num_malic for atk in _ATTACK_TYPES_CANON}

    # Stream CSV
    for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunksize):
        chunk.columns = _normalize_headers(chunk.columns)
        if "Label" not in chunk.columns:
            raise ValueError("Label column not found in CSV")

        labels_norm = chunk["Label"].map(_normalize_label)

        # numeric feature matrix in this chunk (drop strings)
        drop_cols = ["Label"]
        for c in ("Source IP", "Destination IP"):
            if c in chunk.columns:
                drop_cols.append(c)
        Xc = (
            chunk.drop(columns=drop_cols, errors="ignore")
                 .apply(pd.to_numeric, errors="coerce")
                 .fillna(0.0)
                 .to_numpy(dtype=np.float32)
        )

        # Benign
        if need_b > 0:
            idx_b = np.where(labels_norm == "BENIGN")[0]
            if len(idx_b) > 0:
                take_b = min(need_b, len(idx_b))
                pick_b = rng.choice(idx_b, size=take_b, replace=False)
                Xb_parts.append(Xc[pick_b])
                need_b -= take_b

        # Attacks needed for first Na action nodes
        for atk in _ATTACK_TYPES_CANON[:Na]:
            if need_attacks[atk] <= 0:
                continue
            idx_a = np.where(labels_norm == atk)[0]
            if len(idx_a) == 0:
                continue
            take_a = min(need_attacks[atk], len(idx_a))
            pick_a = rng.choice(idx_a, size=take_a, replace=False)
            attack_to_parts[atk].append(Xc[pick_a])
            need_attacks[atk] -= take_a

        if need_b <= 0 and all(v <= 0 for v in need_attacks.values()):
            break

    # Sanity checks
    if need_b > 0:
        raise ValueError(f"Not enough BENIGN rows; still need {need_b}.")
    for atk in _ATTACK_TYPES_CANON[:Na]:
        if need_attacks[atk] > 0:
            raise ValueError(f"Not enough '{atk}' rows; still need {need_attacks[atk]}.")

    # Stack unscaled selections
    X_benign_all = np.concatenate(Xb_parts, axis=0)             # [total_benign_needed, F_raw]
    attacks_arrays = []
    for atk in _ATTACK_TYPES_CANON[:Na]:
        arr = np.concatenate(attack_to_parts[atk], axis=0)       # [>=num_malic, F_raw]
        attacks_arrays.append(arr[:num_malic])

    # Scale on combined set (stable ranges)
    X_stack = np.vstack([X_benign_all] + attacks_arrays).astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_stack = scaler.fit_transform(X_stack).astype(np.float32)

    # Split back
    b_rows = X_benign_all.shape[0]
    X_benign_all = X_stack[:b_rows]
    pos = b_rows
    attacks_arrays_scaled = []
    for arr in attacks_arrays:
        attacks_arrays_scaled.append(X_stack[pos: pos + arr.shape[0]])
        pos += arr.shape[0]

    # Force SAME width for benign & attacks: target_F = Na * Fp_target
    def fix_cols(A: np.ndarray, target_cols: int) -> np.ndarray:
        if A.shape[1] < target_cols:
            return np.pad(A, ((0, 0), (0, target_cols - A.shape[1])), mode="constant")
        elif A.shape[1] > target_cols:
            return A[:, :target_cols]
        return A

    X_benign_all = fix_cols(X_benign_all, target_F)
    for i in range(len(attacks_arrays_scaled)):
        attacks_arrays_scaled[i] = fix_cols(attacks_arrays_scaled[i], target_F)

    # Reshape benign into [*, Na, Fp_target]
    benign_3d = X_benign_all.reshape(-1, Na, Fp_target)   # SAFE now

    # Outputs: benign
    x_benign = torch.from_numpy(benign_3d[:num_benign]).float()
    y_benign = torch.zeros((num_benign, Na), dtype=torch.long)

    # Base for malicious (next num_malic*Na benign slices)
    x_malic = torch.from_numpy(benign_3d[num_benign : num_benign + num_malic * Na]).float()
    y_malic = torch.zeros((num_malic * Na, Na), dtype=torch.long)

    # Inject each attack into its action-node column (slice Fp_target block; NO reshape)
    for idx, _val in enumerate(action_node_idx):
        arr = attacks_arrays_scaled[idx]                    # [num_malic, target_F]
        start = idx * Fp_target
        end   = (idx + 1) * Fp_target
        node_slice = arr[:, start:end]                      # [num_malic, Fp_target]
        beg = idx * num_malic
        end_b = (idx + 1) * num_malic
        x_malic[beg:end_b, idx, :] = torch.from_numpy(node_slice).float()
        y_malic[beg:end_b, idx] = 1

    return x_benign, y_benign, x_malic, y_malic

# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrapper for your existing "dataset2" experiments (unchanged API)
# ──────────────────────────────────────────────────────────────────────────────
def gene_dataset(action_node_idx, num_nodes, num_benign, num_malic):
    """
    Generate Dataset 2 (keeps downstream shapes).
    """
    num_action_nodes = len(action_node_idx)
    x_benign, y_benign, x_malic, y_malic = load_CICIDS(num_benign, num_malic, action_node_idx)

    rt_meas_dim = x_benign.shape[2]
    X_benign = torch.zeros(num_benign, num_nodes, rt_meas_dim)
    X_benign[:, action_node_idx, :] = x_benign
    Y_benign = y_benign

    X_malic = torch.zeros(num_malic * num_action_nodes, num_nodes, rt_meas_dim)
    X_malic[:, action_node_idx, :] = x_malic
    Y_malic = y_malic

    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)
    return X, Y

# ──────────────────────────────────────────────────────────────────────────────
# NEW: LM-only edge dataset loader from CICIDS flows (for LM specialization)
# ──────────────────────────────────────────────────────────────────────────────
from .lm_edges import build_from_ip_port_flow_csv, LMEdgeData  # type: ignore

def load_LM_from_CICIDS(
    csv_path: str = "../datasets/public/CICIDS-2017.csv",
    time_col: str | None = None,            # set to "Timestamp" if present
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> "LMEdgeData":
    """
    LM-only edge dataset:
      - x: node features (hosts)
      - edge_index: directed host→host edges
      - edge_attr: per-edge features (ports/flags/timing)
      - y_edge: {0,1} (LM if attack & LM-port, else 0; benign+LM-port+handshake also marked 1)
      - train/val/test masks (time-ordered)
    """
    return build_from_ip_port_flow_csv(
        csv_path=csv_path,
        src_ip_col="Source IP",
        dst_ip_col="Destination IP",
        src_port_col="Source Port",
        dst_port_col="Destination Port",
        label_col="Label",
        time_col=time_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
