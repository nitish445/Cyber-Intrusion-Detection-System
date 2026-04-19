import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from utils.graph_utils import build_user_graph

print("📡 Building graph...", flush=True)

data = build_user_graph(
    "processed/user_daily.csv",
    "processed/anomaly_scores.csv"
)

print("📦 Graph ready", flush=True)

#DEFINE GNN MODEL
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNN()

#TRAINING SETUP
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

#GNN
print("🚀 Training GNN...", flush=True)

for epoch in range(50):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()

    #if epoch % 10 == 0:
    #    print(f"Epoch {epoch} | Loss: {loss.item():.4f}", flush=True)

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

#SAVE MODEL
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gnn_model.pth")

print("💾 GNN model saved to models/gnn_model.pth", flush=True)
