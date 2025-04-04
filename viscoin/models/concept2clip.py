""" 
Model to convert concept space embeddings to clip embeddings
"""

import torch
import torch.nn as nn


class Concept2CLIP(nn.Module):
    """Basic adapter model with two linear layers"""

    def __init__(self, in_features: int, out_features: int, hidden_size: int = 1024):
        """
        Args:
            in_features (int): dimension of the concept embeddings
            out_features (int): dimension of the clip embeddings
            hidden_size (int, optional): Defaults to 1024.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)
