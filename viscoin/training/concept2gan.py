"""
Train the Concept2GAN model pipeline.
"""

import itertools
import json

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.datasets.utils import (
    DatasetType,
    MergedDataset,
    get_datasets,
)
from viscoin.models.classifiers import Classifier
from viscoin.models.concept2gan import Concept2GAN
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.hyperstyle import compute_w_plus_for_dataset
from viscoin.testing.concept2gan import test_concept2gan
from viscoin.training.losses import (
    concept_orthogonality_loss,
    concept_regularization_loss,
)
from viscoin.utils.dataclasses import IgnoreNone
from viscoin.utils.logging import get_logger
from viscoin.utils.metrics import logits_accuracy


class Concept2GANTrainingParams(IgnoreNone):
    """Concept2GAN training parameters, adapted from VisCoIN CelebA-HQ training."""

    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 8

    # Params adapted from VisCoIN CelebA-HQ training
    of_weight: float = 0.5  # Output fidelity loss weight
    cr_weight: float = 2  # Sparsity loss
    w_weight: float = 3  # W+ space reconstruction loss weight


def train_concept2gan(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    concept2gan: Concept2GAN,
    explainer: Explainer,
    dataset: DatasetType,
    device: str,
    params: Concept2GANTrainingParams,
):

    # Prepare the datasets and precomputed W+ spaces
    w_train, w_test = compute_w_plus_for_dataset(dataset, 4)
    img_train, img_test = get_datasets(dataset)

    train_merged = MergedDataset(img_train, w_train)
    test_merged = MergedDataset(img_test, w_test)

    train = DataLoader(train_merged, batch_size=params.batch_size, shuffle=True)
    test = DataLoader(test_merged, batch_size=params.batch_size, shuffle=False)

    optimizer = optim.Adam(
        itertools.chain(
            concept2gan.parameters(),
            concept_extractor.parameters(),
        ),
        lr=params.learning_rate,
    )

    logger = get_logger()

    classifier.eval()

    for _ in tqdm(range(params.epochs), desc="Training Concept2GAN pipeline"):

        concept_extractor.train()
        concept2gan.train()
        explainer.train()

        # Losses aggregated over batches for logging
        total_accuracy = 0
        total_of = 0
        total_cr = 0
        total_co = 0
        total_w = 0

        for images, _, w_plus in train:

            # Frozen classifier pass
            with torch.no_grad():
                preds, latents = classifier.forward(images.to(device))

            # Extract concepts
            concepts, _ = concept_extractor.forward(latents[-3:])
            # Rebuild predictions from concepts
            concept_preds = explainer.forward(concepts)
            # Rebuild W+ code from concepts
            w_plus_rebuilt = concept2gan.forward(concepts)

            # Compute losses
            output_fidelity = F.cross_entropy(concept_preds, preds) * params.of_weight
            concept_regularization = concept_regularization_loss(concepts) * params.cr_weight
            concept_orthogonality = concept_orthogonality_loss(concept_extractor)
            w_loss = F.mse_loss(w_plus_rebuilt, w_plus.to(device)) * params.w_weight
            total_accuracy += logits_accuracy(concept_preds, preds)

            total = output_fidelity + concept_regularization + concept_orthogonality + w_loss

            total.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Aggregate losses for logging
            total_of += output_fidelity.item()
            total_cr += concept_regularization.item()
            total_co += concept_orthogonality.item()
            total_w += w_loss.item()

        data = test_concept2gan(
            classifier, concept_extractor, concept2gan, explainer, test, device, verbose=False
        )

        data = data | {
            "train_accuracy": total_accuracy / len(train),
            "output_fidelity": total_of / len(train),
            "concept_regularization": total_cr / len(train),
            "concept_orthogonality": total_co / len(train),
            "w_loss": total_w / len(train),
        }
        logger.info(json.dumps(data))
