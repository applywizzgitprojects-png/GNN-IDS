import numpy as np
import torch
import os
import random

def set_seed(seed: int = 40) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

def benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim):
    """ Generate benign data
    """
    set_seed()
    action_node_idx = list(action_nodes.keys())
    num_action_nodes = len(action_node_idx)

    X = torch.zeros(num_samples, num_nodes, rt_meas_dim)
    Y = torch.zeros(num_samples, num_nodes, dtype=torch.float32)
    sd = 0.2
    rt_measurements = []
    for i in range(rt_meas_dim//3):
        mu = np.random.uniform(0.3, 0.3)
        lambda_p = np.random.uniform(3.0, 3.0)
        rt_1 = torch.normal(mu, sd, size=(num_samples, num_action_nodes))
        rt_2 = torch.poisson(torch.ones(num_samples, num_action_nodes)*lambda_p)
        rt_3 = rt_1.abs() ** 0.5 + rt_2 * .5 + 0.5
        rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
    
    rt_measurements = torch.cat(rt_measurements, dim=2)
    X[:, action_node_idx, :] = rt_measurements

    return X, Y

def malic_data(num_samples, num_nodes, action_nodes, rt_meas_dim):
    """ Generate malicious data
    """
    action_node_idx = list(action_nodes.keys())

    comp_node = [[i] for i in action_node_idx]
    num_comp_scenarios = len(comp_node)
    X = torch.zeros(num_samples*num_comp_scenarios, num_nodes, rt_meas_dim)
    Y = torch.zeros(num_samples*num_comp_scenarios, num_nodes, dtype=torch.float32)

    for idx, scenario in enumerate(comp_node):
        num_comp_nodes = len(scenario)
        x, y = benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim)
        action_name = action_nodes[scenario[0]]['predicate']

        mali_meas = sample_mali_scen(num_samples, num_comp_nodes, rt_meas_dim, action_name)

        x[:, scenario, :] = mali_meas
        y[:, scenario] = 1

        X[idx*num_samples:(idx+1)*num_samples, :, :] = x
        Y[idx*num_samples:(idx+1)*num_samples, :] = y

    return X, Y


def sample_mali_scen(num_samples, num_comp_nodes, rt_meas_dim, action_name):
    """ Sample malicious measurements for a scenario
    """
    set_seed()
    rt_measurements = []
    if 'access' in action_name.lower():
        sd = 0.2
        for i in range(rt_meas_dim//3):
            mu = np.random.uniform(0.3, 0.3)
            lambda_p = np.random.uniform(1, 5)
            rt_1 = torch.normal(mu, sd, size=(num_samples, num_comp_nodes))
            rt_2 = torch.poisson(torch.ones(num_samples, num_comp_nodes)*lambda_p)
            rt_3 = rt_1.abs() ** 0.5 + rt_2 * .5 + 0.5
            rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))

    elif 'execcode' in action_name.lower():
        sd = 0.1
        for i in range(rt_meas_dim//3):
            mu = np.random.uniform(0.1, 0.5)
            lambda_p = np.random.uniform(3.0, 3.0)
            rt_1 = torch.normal(mu, sd, size=(num_samples, num_comp_nodes))
            rt_2 = torch.poisson(torch.ones(num_samples, num_comp_nodes)*lambda_p)
            rt_3 = rt_1.abs() ** 0.5 + rt_2 * .5 + 0.5

            rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
    else:
        raise ValueError('event_type should be one of access, execute')
    return torch.cat(rt_measurements, dim=2)

def gene_dataset(num_benign, num_malic, num_nodes, action_nodes, rt_meas_dim):
    """ Generate  Dataset 1
    """
    X_benign, Y_benign = benign_data(num_benign, num_nodes, action_nodes, rt_meas_dim)
    X_malic, Y_malic   = malic_data(num_malic, num_nodes, action_nodes, rt_meas_dim)

    action_mask = list(action_nodes.keys())

    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)[:, action_mask]

    return X, Y

# ──────────────────────────────────────────────────────────────────────────────
# NEW: tiny synthetic LM edge dataset (host pivot chains) — quick sanity tests
# ──────────────────────────────────────────────────────────────────────────────
class LMSynthEdgeData:
    def __init__(self, x, edge_index, edge_attr, y_edge, train_mask_edge, val_mask_edge, test_mask_edge):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y_edge = y_edge
        self.train_mask_edge = train_mask_edge
        self.val_mask_edge = val_mask_edge
        self.test_mask_edge = test_mask_edge

def make_synth_lm_edges(
    num_hosts=50,
    num_pos_edges=800,
    num_neg_edges=3200,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=1234,
) -> LMSynthEdgeData:
    rng = np.random.default_rng(seed)

    # Node features: toy 16-dim zeros
    x = torch.zeros((num_hosts, 16), dtype=torch.float32)

    # Positive LM-ish edges: simple chains A->B (neighbors) with LM ports encoded in attr[0]
    pos_src = rng.integers(0, num_hosts-2, size=num_pos_edges, dtype=np.int64)
    pos_dst = pos_src + 1
    pos_attr = np.column_stack([
        rng.choice([3389, 445, 5985, 22], size=num_pos_edges),  # LM-ish dst ports
        rng.integers(0, 24, size=num_pos_edges),
        rng.integers(0, 2, size=num_pos_edges),
        rng.integers(0, 2, size=num_pos_edges),
    ]).astype(np.float32)

    # Negatives: random pairs not close neighbors, non-LM ports
    neg_src = rng.integers(0, num_hosts, size=num_neg_edges, dtype=np.int64)
    neg_dst = rng.integers(0, num_hosts, size=num_neg_edges, dtype=np.int64)
    mask = (np.abs(neg_src - neg_dst) > 3) & (neg_src != neg_dst)
    neg_src, neg_dst = neg_src[mask], neg_dst[mask]
    neg_attr = np.column_stack([
        rng.choice([80, 53, 123, 25], size=len(neg_src)),
        rng.integers(0, 24, size=len(neg_src)),
        rng.integers(0, 2, size=len(neg_src)),
        rng.integers(0, 2, size=len(neg_src)),
    ]).astype(np.float32)

    src = np.concatenate([pos_src, neg_src])
    dst = np.concatenate([pos_dst, neg_dst])
    edge_attr = torch.tensor(np.vstack([pos_attr, neg_attr]), dtype=torch.float32)
    y_edge = torch.tensor(
        np.concatenate([np.ones(len(pos_src), dtype=np.int64), np.zeros(len(neg_src), dtype=np.int64)]),
        dtype=torch.long
    )
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    # time-ordered split by construction index
    E = edge_index.shape[1]
    e_train = int(E * train_ratio)
    e_val = int(E * (train_ratio + val_ratio))
    train_mask = torch.zeros(E, dtype=torch.bool); train_mask[:e_train] = True
    val_mask = torch.zeros(E, dtype=torch.bool); val_mask[e_train:e_val] = True
    test_mask = torch.zeros(E, dtype=torch.bool); test_mask[e_val:] = True

    return LMSynthEdgeData(x, edge_index, edge_attr, y_edge, train_mask, val_mask, test_mask)
