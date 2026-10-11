import torch.nn as nn


class LSTMModel(nn.Module):
    """Recurrent Neural Network for sequential time-series modeling"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Reshape to (batch, seq_len=1, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
