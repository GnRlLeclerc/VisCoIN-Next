"""
CLI entrypoint for interacting with the viscoin library.

Example usage:
```bash
python main.py test resnet50 --batch-size 32 --dataset-path datasets/CUB_200_2011/
```

Will be subject to many changes as the project evolves.
"""

import click
import numpy.random as rd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from viscoin.datasets.cub import CUB_200_2011
from viscoin.models import explainers
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.utils import load_viscoin, load_viscoin_pickle
from viscoin.testing.classifiers import test_classifier
from viscoin.testing.viscoin import amplify_concepts, plot_amplified_images_batch
from viscoin.training.classifiers import train_classifier_cub
from viscoin.training.viscoin import TrainingParameters, train_viscoin_cub
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
    click.option("--checkpoints", help="The path to load the checkpoints", type=str)(func)

    return func


@main.command()
@common_params
@click.option(
    "--epochs",
    help="The amount of epochs to train the model for",
    default=30,
    type=int,
)
@click.option(
    "--learning-rate",
    help="The optimizer learning rate",
    default=0.0001,
    type=float,
)
@click.option(
    "--output-weights",
    help="The path/filename where to save the weights",
    type=str,
    default="output-weights.pt",
)
@click.option(
    "--gradient-accumulation-steps",
    help="The amount of steps to accumulate gradients before stepping the optimizers",
    type=int,
    default=1,
)
def train(
    model_name: str,
    batch_size: int,
    device: str,
    dataset_path: str,
    checkpoints: str | None,
    epochs: int,
    learning_rate: float,
    output_weights: str,
    gradient_accumulation_steps: int,
):
    """Train a model on a dataset.

    A progress bar is displayed during training.
    Metrics are logged to a file.
    """
    train_dataset = CUB_200_2011(dataset_path, mode="train")
    test_dataset = CUB_200_2011(dataset_path, mode="test")
    batch_size = batch_size // gradient_accumulation_steps
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pretrained = checkpoints is not None

    match model_name:
        case "classifier":
            model = Classifier(output_classes=200, pretrained=pretrained).to(device)

            if pretrained:
                model.load_state_dict(torch.load(checkpoints, weights_only=True))

            model = model.to(device)

            configure_score_logging(f"{model_name}_{epochs}.log")
            train_classifier_cub(
                model,
                train_loader,
                test_loader,
                device,
                epochs,
                learning_rate,
            )

            weights = model.state_dict()
            torch.save(weights, output_weights)

        case "viscoin":
            classifier = torch.load("checkpoints/cub/classifier-cub.pkl", weights_only=False).to(
                device
            )
            concept_extractor = ConceptExtractor().to(device)
            explainer = explainers.Explainer().to(device)
            viscoin_gan = torch.load("checkpoints/cub/gan-adapted-cub.pkl", weights_only=False).to(
                device
            )
            generator_gan = torch.load("checkpoints/cub/gan-cub.pkl", weights_only=False).to(device)

            if pretrained:
                load_viscoin(classifier, concept_extractor, explainer, viscoin_gan, checkpoints)

            configure_score_logging(f"{model_name}_{epochs}.log")

            # Using the default parameters for training on CUB
            params = TrainingParameters()
            params.gradient_accumulation = gradient_accumulation_steps

            # The training saves the viscoin model regularly
            train_viscoin_cub(
                classifier,
                concept_extractor,
                explainer,
                viscoin_gan,
                generator_gan,
                train_loader,
                test_loader,
                params,
                device,
            )

        case _:
            raise ValueError(f"Unknown model name: {model_name}")


@main.command()
@common_params
def test(
    model_name: str,
    batch_size: int,
    device: str,
    dataset_path: str,
    checkpoints: str | None,
):
    """Test a model on a dataset"""

    dataset = CUB_200_2011(dataset_path, mode="test")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pretrained = checkpoints is not None

    match model_name:
        case "classifier":
            model = Classifier(output_classes=200, pretrained=pretrained).to(device)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    if pretrained:
        model.load_state_dict(torch.load(checkpoints, weights_only=True))

    # TODO : if viscoin, specific stuff has to be done to ensure the model parameters are compatible

    model = model.to(device)

    accuracy, loss = test_classifier(model, dataloader, device)

    click.echo(f"Accuracy: {100*accuracy:.2f}%")
    click.echo(f"Loss: {loss}")


@main.command()
@click.option(
    "--dataset-path",
    help="The path to the dataset to use for training/testing",
    required=True,
    type=str,
)
@click.option(
    "--viscoin-pickle-path",
    help="The path to the viscoin pickle file",
    required=True,
    type=str,
)
@click.option(
    "--n-samples",
    help="The number of random images to amplify",
    type=int,
    default=5,
)
@click.option(
    "--device",
    help="The device to use for training/testing",
    type=str,
    default="cuda",
)
def amplify(
    dataset_path: str,
    viscoin_pickle_path: str,
    n_samples: int,
    device: str,
):
    """Amplify the concepts of random images from a dataset (showcase)"""
    # Load dataset and models
    models = load_viscoin_pickle(viscoin_pickle_path)
    dataset = CUB_200_2011(dataset_path, mode="test")

    # Move models to device
    classifier = models.classifier.to(device)
    concept_extractor = models.concept_extractor.to(device)
    explainer = models.explainer.to(device)
    gan = models.gan.to(device)

    # Choose random images to amplify
    indices = rd.choice(len(dataset), n_samples, replace=False)
    originals = [dataset[i][0].to(device) for i in indices]
    amplified: list[list[Tensor]] = []
    multipliers: list[float] = []

    for image in originals:
        results = amplify_concepts(image, classifier, concept_extractor, explainer, gan, device)
        amplified.append(results.amplified_images)
        multipliers = results.multipliers

    plot_amplified_images_batch(originals, amplified, multipliers)


if __name__ == "__main__":
    main()
