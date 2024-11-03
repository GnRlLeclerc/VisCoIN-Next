"""
CLI entrypoint for interacting with the viscoin library.

Example usage:
```bash
python main.py test resnet50 --batch-size 32 --dataset-path datasets/CUB_200_2011/
```
"""

import click
from torch.utils.data import DataLoader

from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.classifiers import Classifier
from viscoin.testing.classifiers import test_classifier


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
def train():
    raise NotImplementedError("Training command not implemented yet")


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

    # TODO : if not pretrained, load the classifier checkpoints. Add helper methods for this
    # TODO : if viscoin, specific stuff has to be done to ensure the model parameters are compatible

    model = model.to(device)

    accuracy, loss = test_classifier(model, dataloader, device)

    click.echo(f"Accuracy: {100*accuracy:.2f}%")
    click.echo(f"Loss: {loss}")


if __name__ == "__main__":
    main()
