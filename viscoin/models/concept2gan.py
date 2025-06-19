"""From 16 x 3 x 3 VisCoIN concept space to 14 x 512 StyleGAN W+ space."""

import torch.nn as nn
from torch import Tensor


class Concept2GAN(nn.Module):
    """Concept2GAN model with two linear layers"""

    def __init__(self, n_concepts: int, style_layers=14, w_dim=512):
        """
        Args:
            n_concepts: amount of viscoin concepts
            style_layers: amount of StyleGAN style layers (default: 14)
        """
        super().__init__()

        hidden_dim = (n_concepts * 3 * 3 + style_layers * w_dim) // 2
        self.style_layers = style_layers
        self.w_dim = w_dim

        self.model = nn.Sequential(
            nn.Linear(n_concepts * 3 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, style_layers * w_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: (batch_size, n_concepts, 3, 3): Concept embeddings
        """
        # Flatten the input tensor to (batch_size, n_concepts * 3 * 3)
        x = x.view(x.size(0), -1)
        x = self.model(x)

        # Reshape to (batch_size, style_layers, w_dim) for comparison with StyleGAN W+ space
        return x.view(-1, self.style_layers, self.w_dim)
