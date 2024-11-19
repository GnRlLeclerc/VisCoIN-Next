"""
CLI entrypoint for interacting with the viscoin library.

Example usage:
```bash
python main.py test resnet50 --batch-size 32 --dataset-path datasets/CUB_200_2011/
```

Will be subject to many changes as the project evolves.
"""

import click
import torch
from torch.utils.data import DataLoader

from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.classifiers import Classifier
from viscoin.testing.classifiers import test_classifier
from viscoin.training.classifiers import train_classifier
from viscoin.utils.logging import configure_score_logging


@click.group()
def main():
    pass


def common_params(func):
    """Add common parameters to the cli command"""
    click.argument("model_name")(func)
    click.option(
        "--batch-size", default=32, help="The batch size to use for training/testing", type=int
    )(func)
    click.option(
        "--device", default="cuda", help="The device to use for training/testing", type=str
    )(func)
    click.option(
        "--dataset-path",
        help="The path to the dataset to use for training/testing",
        required=True,
        type=str,
    )(func)
    click.option(
        "--classifier-checkpoints", help="The path to load the classifier checkpoints", type=str
    )(func)

    return func


@main.command()
@common_params
@click.option(
    "--epochs",
    help="The amount of epochs to train the model for",
    default=160,
    type=int,
)
@click.option(
    "--learning-rate",
    help="The optimizer learning rate",
    default=0.001,
    type=float,
)
@click.option(
    "--sgd_momentum",
    help="The SGD optimizer momentum",
    default=0.9,
    type=float,
)
@click.option(
    "--sgd_weight_decay",
    help="The SGD optimizer weight decay",
    default=1e-4,
    type=float,
)
@click.option(
    "--output-weights",
    help="The path/filename where to save the weights",
    required=True,
    type=str,
)
def train(
    model_name: str,
    batch_size: int,
    device: str,
    dataset_path: str,
    classifier_checkpoints: str | None,
    epochs: int,
    learning_rate: float,
    sgd_momentum: float,
    sgd_weight_decay: float,
    output_weights: str,
):
    """Train a model on a dataset.

    A progress bar is displayed during training.
    Metrics are logged to a file.
    """
    train_dataset = CUB_200_2011(dataset_path, mode="train")
    test_dataset = CUB_200_2011(dataset_path, mode="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pretrained = classifier_checkpoints is None

    match model_name:
        case "resnet18":
            model = Classifier(resnet="18", output_classes=200, pretrained=pretrained).to(device)
        case "resnet50":
            model = Classifier(resnet="50", output_classes=200, pretrained=pretrained).to(device)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    if not pretrained:
        model.load_state_dict(torch.load(classifier_checkpoints, weights_only=True))

    model = model.to(device)

    configure_score_logging(f"{model_name}_{epochs}.log")
    train_classifier(
        model,
        train_loader,
        test_loader,
        device,
        epochs,
        learning_rate,
        sgd_momentum,
        sgd_weight_decay,
    )

    weights = model.state_dict()
    torch.save(weights, output_weights)


@main.command()
@common_params
def test(
    model_name: str,
    batch_size: int,
    device: str,
    dataset_path: str,
    classifier_checkpoints: str | None,
):
    """Test a model on a dataset"""

    dataset = CUB_200_2011(dataset_path, mode="test")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pretrained = classifier_checkpoints is None

    match model_name:
        case "resnet18":
            model = Classifier(resnet="18", output_classes=200, pretrained=pretrained).to(device)
        case "resnet50":
            model = Classifier(resnet="50", output_classes=200, pretrained=pretrained).to(device)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    if not pretrained:
        model.load_state_dict(torch.load(classifier_checkpoints, weights_only=True))

    # TODO : if viscoin, specific stuff has to be done to ensure the model parameters are compatible

    model = model.to(device)

    accuracy, loss = test_classifier(model, dataloader, device)

    click.echo(f"Accuracy: {100*accuracy:.2f}%")
    click.echo(f"Loss: {loss}")


if __name__ == "__main__":
    main()
