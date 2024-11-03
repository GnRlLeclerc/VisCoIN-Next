"""Learning rate schedulers"""

from torch.optim import Optimizer


class CustomLRScheduler:
    """Schedule learning rates by specifying (epochs, learning rates) pairs
    to change the learning rate multiple times after a custom amount of epochs.
    """

    def __init__(self, optimizer: Optimizer, milestones: list[tuple[int, float]]):
        """Initialize the scheduler.

        Args:
            optimizer: the optimizer to update the learning rate of
            milestones: a list of (epoch, lr) tuples to change the learning rate at each epoch
        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.current_epoch = 0

    def step(self):
        # Check if the current epoch is in the milestones (naive method is enough)
        for epoch, lr in self.milestones:
            if self.current_epoch == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

        self.current_epoch += 1
