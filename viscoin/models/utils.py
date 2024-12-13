"""Model utilities."""

from dataclasses import dataclass

import torch

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
