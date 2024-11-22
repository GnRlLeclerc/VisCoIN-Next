"""Utility functions for training."""

from torch.optim import Optimizer


def update_lr(optimizer: Optimizer, lr: float):
    """Update the learning rate of an optimizer"""

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
