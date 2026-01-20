import pandas as pd
import torch
from torch_geometric.data import Data


def build_user_graph(user_daily_path, anomaly_path):
    """
    FAST graph:
    Nodes = users
    Edge = self-loop (required for GCN)
    Labels = anomaly ratio
    """

    print("📂 Loading CSV files...", flush=True)
    user_df = pd.read_csv(user_daily_path)
    anomaly_df = pd.read_csv(anomaly_path)

    min_len = min(len(user_df), len(anomaly_df))
    user_df = user_df.iloc[:min_len].copy()
    anomaly_df = anomaly_df.iloc[:min_len]

    user_df["anomaly"] = anomaly_df["anomaly"].values

    print("🔢 Creating user nodes...", flush=True)
    users = user_df["user"].unique()
    node_map = {u: i for i, u in enumerate(users)}

    # SELF-LOOP GRAPH (FAST & SAFE)
    edges = []
    for u in users:
        idx = node_map[u]
        edges.append([idx, idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Node features
    x = torch.ones((len(users), 1))

    # Labels based on anomaly ratio
    print("🏷️ Assigning labels...", flush=True)
    user_anomaly = user_df.groupby("user")["anomaly"].mean()

    y = torch.zeros(len(users), dtype=torch.long)
    for user, score in user_anomaly.items():
        if score > 0.3:
            y[node_map[user]] = 1

    print("📦 Graph construction complete", flush=True)
    return Data(x=x, edge_index=edge_index, y=y)
