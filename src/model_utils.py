import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# ======================================================
# Train one epoch of LM edge classification
# ======================================================
def train_lm_epoch(backbone, lm_head, data, optimizer, device="cpu"):
    backbone.train(); lm_head.train()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y_edge = data.y_edge.to(device).float()
    edge_attr = (data.edge_attr.to(device)
                 if getattr(data, "edge_attr", None) is not None else None)
    mask = (data.train_mask_edge.to(device)
            if getattr(data, "train_mask_edge", None) is not None
            else torch.ones_like(y_edge).bool())

    # Encode nodes → classify edges
    z = backbone.encode(x, edge_index)
    src, dst = edge_index[0], edge_index[1]
    logits = lm_head(z[src][mask], z[dst][mask],
                     edge_attr[mask] if edge_attr is not None else None)

    # Handle imbalance with pos_weight
    pos = (y_edge[mask] == 1).sum().item()
    neg = (y_edge[mask] == 0).sum().item()
    pos_weight = torch.tensor([max(1.0, neg / max(1, pos))], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    loss = criterion(logits, y_edge[mask])

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return float(loss.item())


# ======================================================
# Full training loop for LM edge classification
# ======================================================
def train_lm(backbone, lm_head, data, lr=1e-3, epochs=10,
             device="cpu", verbose=True):
    params = list(backbone.parameters()) + list(lm_head.parameters())
    optimizer = optim.Adam(params, lr=lr)
    history = {"loss": [], "val_auprc": [], "val_f1": []}

    for ep in range(epochs):
        loss = train_lm_epoch(backbone, lm_head, data, optimizer, device)
        history["loss"].append(loss)

        logits, y_edge = infer_lm(backbone, lm_head, data, device=device)
        if hasattr(data, "val_mask_edge") and data.val_mask_edge is not None:
            mask = data.val_mask_edge.to(device).bool()
            metrics = eval_lm_edge(logits[mask], y_edge[mask], thresh=0.5)
        else:
            metrics = eval_lm_edge(logits, y_edge, thresh=0.5)

        history["val_auprc"].append(metrics["auprc"])
        history["val_f1"].append(metrics["f1"])

        if verbose:
            print(f"[LM] Epoch {ep+1:03d}: "
                  f"loss={loss:.4f} AUPRC={metrics['auprc']:.4f} "
                  f"F1={metrics['f1']:.4f}")
    return history


# ======================================================
# Inference: get logits for all edges
# ======================================================
@torch.no_grad()
def infer_lm(backbone, lm_head, data, device="cpu"):
    backbone.eval(); lm_head.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = (data.edge_attr.to(device)
                 if getattr(data, "edge_attr", None) is not None else None)
    y_edge = data.y_edge.to(device).long() if hasattr(data, "y_edge") else None

    z = backbone.encode(x, edge_index)
    src, dst = edge_index[0], edge_index[1]
    logits = lm_head(z[src], z[dst], edge_attr)
    return logits, y_edge


# ======================================================
# Evaluation: confusion matrix + metrics
# ======================================================
@torch.no_grad()
def eval_lm_edge(logits: torch.Tensor, y_true: torch.Tensor, thresh=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = y_true.detach().cpu().numpy().astype(int)
    y_pred = (probs >= thresh).astype(int)

    if y.sum() == 0:
        tn = (y_pred == 0).sum(); fp = (y_pred == 1).sum()
        fn = 0; tp = 0
    else:
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y, probs)
    except ValueError:
        roc_auc = 0.0
    auprc = average_precision_score(y, probs)

    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "auprc": float(auprc),
    }
