"""Model utilities."""

import torch

from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted


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
