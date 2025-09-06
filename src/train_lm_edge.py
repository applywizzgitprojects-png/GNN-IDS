import torch
import torch.nn as nn

from .models import GCN, LMEdgeHead           # you can swap GCN with GAT/GCN_EW
from .public_data import load_LM_from_CICIDS  # real flows from CICIDS with LM labels
# from .synthetic_data import make_synth_lm_edges  # quick sanity test

class BCEWithLogitsFocal(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets.float())
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

def train_epoch(backbone, head, data, optimizer, criterion):
    backbone.train(); head.train()
    z = backbone.encode(data.x, data.edge_index)                  # (N, D)
    z_src = z[data.edge_index[0]]                                 # (E, D)
    z_dst = z[data.edge_index[1]]                                 # (E, D)
    logits = head(z_src, z_dst, data.edge_attr)                   # (E,)

    mask = data.train_mask_edge
    loss = criterion(logits[mask], data.y_edge[mask].float())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def eval_epoch(backbone, head, data, split='val'):
    backbone.eval(); head.eval()
    z = backbone.encode(data.x, data.edge_index)
    z_src = z[data.edge_index[0]]
    z_dst = z[data.edge_index[1]]
    logits = head(z_src, z_dst, data.edge_attr)
    mask = data.val_mask_edge if split == 'val' else data.test_mask_edge
    y = data.y_edge[mask]
    p = torch.sigmoid(logits[mask])
    pred = (p >= 0.5).long()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {'precision': precision, 'recall': recall, 'f1@0.5': f1}

def main():
    # === Choose dataset ===
    data = load_LM_from_CICIDS(csv_path="../datasets/public/CICIDS-2017.csv", time_col=None)
    # data = make_synth_lm_edges()  # uncomment for a quick smoke test

    in_dim = data.x.shape[1]
    edge_feat_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 0

    backbone = GCN(in_dim=in_dim, hidden_dim=128, out_dim=64)
    head = LMEdgeHead(node_embed_dim=128, edge_feat_dim=edge_feat_dim, hidden=128)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    criterion = BCEWithLogitsFocal(alpha=0.25, gamma=2.0)  # robust to imbalance

    best_val = -1.0
    for epoch in range(1, 21):
        loss = train_epoch(backbone, head, data, optimizer, criterion)
        val_m = eval_epoch(backbone, head, data, split='val')
        if val_m['f1@0.5'] > best_val:
            best_val = val_m['f1@0.5']
            best_state = {'backbone': backbone.state_dict(), 'head': head.state_dict()}
        print(f"[LM] Epoch {epoch:03d}: loss={loss:.4f} valF1={val_m['f1@0.5']:.4f}")

    # Final test
    backbone.load_state_dict(best_state['backbone'])
    head.load_state_dict(best_state['head'])
    test_m = eval_epoch(backbone, head, data, split='test')
    print("Final LM metrics:", test_m)

if __name__ == "__main__":
    main()
