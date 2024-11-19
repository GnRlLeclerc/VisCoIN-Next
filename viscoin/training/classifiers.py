"""Classifier training functions."""

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
    epochs: int,
    learning_rate: float,
    sgd_momentum: float,
    sgd_weight_decay: float,
):
    """Train the classifier model. The best model on testing data is loaded into the classifier instance.

    Note: the losses are averaged over batches.

    Args:
        model: the classifier model to train
        train_loader: the DataLoader containing the training dataset
        test_loader: the DataLoader containing the testing dataset
        device: the device to use for training
        epochs: the number of epochs to train the model
        learning_rate: the learning rate for the SGD optimizer
        sgd_momentum: the momentum for the SGD optimizer
        sgd_weight_decay: the weight decay for the SGD optimizer
    """
    test_loss: list[float] = []
    train_loss: list[float] = []
    train_accuracy: list[float] = []
    test_accuracy: list[float] = []
    best_accuracy = 0.0

    best_model = model.state_dict()
    logger = get_logger()

    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=sgd_momentum, weight_decay=sgd_weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # Fine-tuning scheduler
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
            preds = outputs.argmax(dim=1, keepdim=True)

            # Compute loss and backpropagate
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Update training metrics
            total_correct += preds.eq(targets.view_as(preds)).sum().item()
            total_loss += loss.item()
            total_samples += targets.size(0)

        # Append training metrics
        accuracy = 100 * total_correct / total_samples
        batch_mean_loss = total_loss / len(train_loader)
        train_loss.append(batch_mean_loss)
        train_accuracy.append(accuracy)
        scheduler.step()

        ###########################################################################################
        #                                       TESTING STEP                                      #
        ###########################################################################################

        accuracy, mean_loss = test_classifier(model, test_loader, device, criterion, False)
        test_loss.append(mean_loss)
        test_accuracy.append(accuracy)

        if accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = accuracy

        # Log the current state of training
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {batch_mean_loss:.4f} - Train Acc: {accuracy:.2f}% - Test Loss: {mean_loss:.4f} - Test Acc: {accuracy:.2f}%"
        )

    # Load the best model
    print(f"Best test accuracy: {best_accuracy:.4f}")
    model.load_state_dict(best_model)
