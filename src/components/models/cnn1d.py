import torch.nn as nn


class CNN1DModel(nn.Module):
    """1D Convolutional Neural Network for time-series feature extraction"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Reshape to (batch, 1, input_dim)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
