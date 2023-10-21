import torch
from torch import nn
from torch.nn import functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self, context_len, n_embed):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(1000, n_embed)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(context_len * n_embed, 512),
            # nn.Linear(context_len, 512),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        if y is None:
            loss = None
        else:
            loss = F.binary_cross_entropy(torch.sigmoid(logits), y)
        return torch.sigmoid(logits), loss
