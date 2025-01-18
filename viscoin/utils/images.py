"""Image utilities"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, clamp


def clip_tensor_image(x: Tensor) -> Tensor:
    """Clip a tensor / batched tensor's values for image display depending on the type.
    - int: [0, 255]
    - float: [0, 1]
    This avoids cluttering stdout with warnings about values needing to be clipped when using matplotlib.
    """
    match x.dtype:
        case torch.float:
            return clamp(x, 0, 1)
        case torch.int | torch.uint8:
            return clamp(x, 0, 255)
        case _:
            raise ValueError(f"Unsupported tensor type {x.dtype}")


def normalize_tensor_image(x: Tensor) -> Tensor:
    """Normalize a tensor / batched tensor's values for image display depending on the type.
    - int: [0, 255] -> [0, 1]
    - float: [0, 1]
    """
    min = x.min()
    max = x.max()

    match x.dtype:
        case torch.float:
            return (x - min) / (max - min)
        case torch.int | torch.uint8:
            return ((x.float() - min) / (max - min) * 255).to(torch.uint8)
        case _:
            raise ValueError(f"Unsupported tensor type {x.dtype}")


def from_torch(x: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor representing images to a numpy array.
    Accepts a batched or unbatched single image."""

    dim = len(x.shape)
    x = normalize_tensor_image(x)

    if dim == 3:
        return x.permute(1, 2, 0).detach().cpu().numpy()
    elif dim == 4:
        return x.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    raise ValueError(f"Unsupported shape for image tensor: {dim}")


def imshow(x: Tensor, title: str):
    """Show a PyTorch tensor representing an image.
    Args:
        x (1, width, height, 3): The image tensor to show, batched or unbatched.
    """

    image = from_torch(x)

    # Remove the batch dimension if it exists
    if len(image.shape) == 4:
        assert image.shape[0] == 1, f"Batch size of {image.shape[0]} invalid, expected 1"
        image = image[0]

    plt.title(title)
    plt.imshow(image)
    plt.show()


def heatmap_to_img(heatmap: np.ndarray):
    """Convert a numpy heatmap to a RGB uint8 image for display or overlay"""
    heatmap = heatmap.squeeze()

    # Normalize the heatmap
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore

    # Apply the usual colormap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Resize to (256, 256)
    heatmap_resized = cv2.resize(heatmap_colored, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Convert BGR to RGB for displaying with Matplotlib
    heatmap_resized_rgb = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)

    return heatmap_resized_rgb


def overlay(image: np.ndarray, overlay: np.ndarray, alpha=0.4):
    """Overlay an overlay image on top of a base image"""
    assert image.dtype == overlay.dtype, "Image and overlay must have the same dtype"

    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
