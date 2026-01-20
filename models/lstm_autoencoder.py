import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(h)
        return out
