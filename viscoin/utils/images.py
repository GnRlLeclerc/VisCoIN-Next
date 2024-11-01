"""Image utilities"""

from torch import Tensor


def from_torch(x: Tensor):
    """Convert a PyTorch tensor representing images to a numpy array"""
    return x.permute(0, 2, 3, 1).detach().cpu().numpy()
