"""Utility functions for training."""

from typing import Iterable, Iterator, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from stylegan2_ada.training.networks import Generator


def update_lr(optimizer: Optimizer, lr: float) -> None:
    """Update the learning rate of an optimizer"""

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def requires_grad(model: nn.Module, flag=True) -> None:
    """Set the requires_grad flag of a model's parameters.
    Useful for freezing specific layers of a model for fine-tuning."""
    for p in model.parameters():
        p.requires_grad = flag


def synthetic_samples(gan: Generator, n_samples: int, device: str) -> Tensor:
    """Take `n_samples` synthetic samples from the GAN.

    Args:
        gan (Generator): The NON-ADAPTED GAN model trained on a specific dataset (CUB for instance).
        n_samples (int): The number of samples to generate.
        device (str): The device to use.

    Returns:
        Tensor (n_samples, 3, 256, 256): The synthetic samples.
    """
    z_samples = torch.randn(n_samples, gan.z_dim)
    samples = gan(z_samples.to(device), None)
    return samples


T = TypeVar("T")


def loop_iter(loader: Iterable[T]) -> Iterator[T]:
    """Transform an iterable into an iterable that loops around indefinitely."""
    while True:
        for batch in loader:
            yield batch


class Accumulator:
    """Gradient accumulation scheduler"""

    def __init__(self, delay_epochs: int) -> None:
        """Create a new gradient accumulation scheduler.
        Upon stepping, it will return `true` every `delay_epochs` epochs."""

        self.delay_epochs = delay_epochs
        self.accumulation_steps = 0

    def step(self) -> bool:
        """Step the scheduler and return `true` if the accumulation should be done."""
        self.accumulation_steps += 1

        if self.accumulation_steps >= self.delay_epochs:
            self.accumulation_steps = 0
            return True
        return False
