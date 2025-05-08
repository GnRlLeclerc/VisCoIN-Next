"""Model utilities."""

import os
from dataclasses import dataclass

import torch
from torch import Tensor
from tqdm import tqdm

from viscoin.datasets.utils import DatasetType, get_dataloaders
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted


@dataclass
class VisCoINModels:
    """Dataclass to store the VisCoIN models together for pickling."""

    classifier: Classifier
    concept_extractor: ConceptExtractor
    explainer: Explainer
    gan: GeneratorAdapted

    def compute_w_space(
        self,
        dataset: DatasetType,
        device: str,
    ) -> tuple[Tensor, Tensor]:
        """Compute and return the train and test W+ spaces of the GAN for the given dataset.
        The results are cached under checkpoints/gan-w/

        Returns:
            train_w: W+ space for the training set
            test_w: W+ space for the testing set
        """

        try:
            return (
                torch.load(f"checkpoints/gan-w/{dataset}-train.pt", weights_only=True),
                torch.load(f"checkpoints/gan-w/{dataset}-test.pt", weights_only=True),
            )
        except FileNotFoundError:
            pass

        gan = self.gan.to(device)
        classifier = self.classifier.to(device)
        concept_extractor = self.concept_extractor.to(device)
        batch_size = 4

        train_loader, test_loader = get_dataloaders(dataset, batch_size, "test")

        train_w = torch.zeros((len(train_loader.dataset), gan.num_ws, gan.w_dim))  # type: ignore
        test_w = torch.zeros((len(test_loader.dataset), gan.num_ws, gan.w_dim))  # type: ignore

        for i, (img, _) in enumerate(
            tqdm(
                train_loader,
                desc=f"Computing W+ space for {dataset} - train",
            )
        ):
            img = img.to(device)

            with torch.no_grad():
                _, latents = classifier.forward(img)
                concepts, extra_info = concept_extractor.forward(latents[-3:])
                _, latents = gan.forward(
                    z1=concepts,
                    z2=extra_info,
                    return_latents=True,
                )

            train_w[i * batch_size : (i + 1) * batch_size] = latents.detach().cpu()

        for i, (img, _) in enumerate(
            tqdm(
                test_loader,
                desc=f"Computing W+ space for {dataset} - test",
            )
        ):
            img = img.to(device)

            with torch.no_grad():
                _, latents = classifier.forward(img)
                concepts, extra_info = concept_extractor.forward(latents[-3:])
                _, latents = gan.forward(
                    z1=concepts,
                    z2=extra_info,
                    return_latents=True,
                )

            test_w[i * batch_size : (i + 1) * batch_size] = latents.detach().cpu()

        # Save the W+ spaces
        os.makedirs("checkpoints/gan-w", exist_ok=True)
        torch.save(train_w, f"checkpoints/gan-w/{dataset}-train.pt")
        torch.save(test_w, f"checkpoints/gan-w/{dataset}-test.pt")

        return train_w, test_w


def save_viscoin(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    gan: GeneratorAdapted,
    path: str,
):
    """Jointly save the checkpoints of the VisCoIN models."""
    checkpoints = {
        "classifier": classifier.state_dict(),
        "concept_extractor": concept_extractor.state_dict(),
        "explainer": explainer.state_dict(),
        "gan": gan.state_dict(),
    }

    torch.save(checkpoints, path)


def load_viscoin(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    gan: GeneratorAdapted,
    path: str,
):
    """Jointly load the checkpoints of the VisCoIN models."""
    checkpoints = torch.load(path, weights_only=True)

    classifier.load_state_dict(checkpoints["classifier"])
    concept_extractor.load_state_dict(checkpoints["concept_extractor"])
    explainer.load_state_dict(checkpoints["explainer"])
    gan.load_state_dict(checkpoints["gan"])


def save_viscoin_pickle(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    gan: GeneratorAdapted,
    path: str,
):
    """Jointly pickle the VisCoIN models to also store their parameters."""
    bundle = VisCoINModels(
        classifier=classifier,
        concept_extractor=concept_extractor,
        explainer=explainer,
        gan=gan,
    )

    torch.save(bundle, path)


def load_viscoin_pickle(
    path: str,
) -> VisCoINModels:
    """Jointly load the VisCoIN models from a pickle file."""
    return torch.load(path, weights_only=False)
