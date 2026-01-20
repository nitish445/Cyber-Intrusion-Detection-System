import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 1. LOAD PROCESSED DATA
print("📥 Loading processed data...")
df = pd.read_csv("processed/user_daily.csv")

# Sort properly (VERY IMPORTANT)
df["day"] = pd.to_datetime(df["day"])
df = df.sort_values(["user", "day"])

# 2. SELECT FEATURES
features = [
    "emails_sent",
    "unique_receivers",
    "avg_size",
    "attachments",
    "avg_content_len"
]

# 3. SCALE FEATURES
print("📏 Scaling features...")
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 4. CREATE SEQUENCES
SEQ_LEN = 7   # 7 days behaviour window
sequences = []

print("⏳ Creating sequences...")
for user in tqdm(df["user"].unique()):
    user_df = df[df["user"] == user]
    data = user_df[features].values

    if len(data) < SEQ_LEN:
        continue

    for i in range(len(data) - SEQ_LEN):
        sequences.append(data[i:i+SEQ_LEN])

X = np.array(sequences)
print("✅ Total sequences:", X.shape)

# 5. CONVERT TO TORCH
X_tensor = torch.tensor(X, dtype=torch.float32)

# 6. DEFINE LSTM AUTOENCODER
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(h)
        return out

model = LSTMAutoencoder(input_dim=X.shape[2])

# 7. TRAIN LSTM
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

EPOCHS = 20
BATCH_SIZE = 64

print("🚀 Training LSTM Autoencoder...")
for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for i in range(0, len(X_tensor), BATCH_SIZE):
        batch = X_tensor[i:i+BATCH_SIZE]

        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

# 8. SAVE MODEL
torch.save(model.state_dict(), "models/lstm_autoencoder.pth")
print("💾 LSTM model saved to models/lstm_autoencoder.pth")

# 9. ANOMALY DETECTION
print("🚨 Detecting anomalies using reconstruction error...")

model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)

# Mean Squared Error per sequence
recon_error = ((reconstructed - X_tensor) ** 2).mean(dim=(1, 2))

# Convert to numpy
recon_error_np = recon_error.numpy()

# Threshold (statistical)
threshold = recon_error_np.mean() + 2 * recon_error_np.std()

print(f"📏 Anomaly threshold: {threshold:.4f}")

# Flag anomalies
anomaly_flags = (recon_error_np > threshold).astype(int)

print(f"🚨 Total anomalies detected: {anomaly_flags.sum()}")

# 10. SAVE ANOMALY SCORES
anomaly_df = pd.DataFrame({
    "reconstruction_error": recon_error_np,
    "anomaly": anomaly_flags
})

anomaly_df.to_csv("processed/anomaly_scores.csv", index=False)

print("💾 Anomaly scores saved to processed/anomaly_scores.csv")
