"""Image utilities"""

from torch import Tensor


def from_torch(x: Tensor):
    """Convert a PyTorch tensor representing images to a numpy array"""

    dim = len(x.shape)

    if dim == 3:
        return x.permute(1, 2, 0).detach().cpu().numpy()
    elif dim == 4:
        return x.permute(0, 2, 3, 1).detach().cpu().numpy()

    raise ValueError(f"Unsupported shape for image tensor: {dim}")
