"""Classifier training functions.

Best parameters:
- Adam optimizer
    - Learning rate: 0.001
    - Weight decay: 1e-4
- Epochs: 90
- LR Scheduler: StepLR(step=30, gamma=0.1)
- batch size: 32
"""

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.testing.classifiers import test_classifier
from viscoin.utils.logging import get_logger


def train_classifier(
    model: Classifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int = 30,
    learning_rate: float = 0.0001,
):
    """Train the classifier model for the CUB or FunnyBirds dataset. The best model on testing data is loaded into the classifier instance.

    Note: the losses are averaged over batches.

    Args:
        model: the classifier model to train
        train_loader: the DataLoader containing the training dataset
        test_loader: the DataLoader containing the testing dataset
        device: the device to use for training
        epochs: the number of epochs to train the model
        learning_rate: the learning rate for the optimizer
    """
    best_accuracy = 0.0
    best_model = model.state_dict()
    logger = get_logger()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, epochs + 1), "Training epochs"):
        ###########################################################################################
        #                                      TRAINING STEP                                      #
        ###########################################################################################

        model.train()

        # Training metrics for this epoch
        total_correct = 0
        total_loss = 0
        total_samples = 0

        for inputs, targets in train_loader:
            # Move batch to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute logits
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            current_loss = loss.item()

            # Compute loss and backpropagate
            loss.backward()
            optimizer.step()

            # Update training metrics
            preds = outputs.argmax(dim=1, keepdim=True)
            total_correct += preds.eq(targets).sum().item()
            total_loss += current_loss
            total_samples += targets.size(0)

        # Append training metrics
        accuracy = total_correct / total_samples
        batch_mean_loss = total_loss / len(train_loader)
        scheduler.step()

        ###########################################################################################
        #                                       TESTING STEP                                      #
        ###########################################################################################

        accuracy, mean_loss = test_classifier(model, test_loader, device, criterion, False)

        if accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = accuracy

        # Log the current state of training
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {batch_mean_loss:.4f} - Train Acc: {100 * accuracy:.2f}% - Test Loss: {mean_loss:.4f} - Test Acc: {100 * accuracy:.2f}%"
        )

    # Load the best model
    print(f"Best test accuracy: {best_accuracy:.4f}")
    model.load_state_dict(best_model)
