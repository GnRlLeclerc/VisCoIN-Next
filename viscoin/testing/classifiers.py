"""Classifiers testing functions"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier


def test_classifier(
    model: Classifier,
    dataloader: DataLoader,
    device: str,
    criterion: nn.Module | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    """Test the classifier performance across a testing Dataloader

    Args:
        model: the classifier model to test
        dataloader: the DataLoader containing the testing dataset
        device: the device to use for the testing
        criterion: the loss function to use for the test (default: nn.CrossEntropyLoss)
        verbose: whether to print the progress bar (default: True)

    Returns:
        accuracy: the accuracy of the model on the testing dataset
        batch_mean_loss: the mean loss per batch on the testing dataset
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        total_correct = 0
        total_loss = 0
        total_samples = 0

        for inputs, targets in tqdm(dataloader, desc="Test batches", disable=not verbose):
            # Move batch to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute logits & predictions
            outputs, _ = model.forward(inputs)
            preds = outputs.argmax(dim=1, keepdim=True)

            # Update metrics
            total_correct += preds.eq(targets.view_as(preds)).sum().item()
            total_loss += criterion(outputs, targets).item()
            total_samples += targets.size(0)

    accuracy = total_correct / total_samples
    batch_mean_loss = total_loss / len(dataloader)

    return accuracy, batch_mean_loss
