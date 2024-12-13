"""Testing concept repartition in viscoin"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer


@dataclass
class ConceptTestResults:
    """Results of the concept test

    Args:
        classifier_accuracy: The accuracy of the classifier model.
        explainer_accuracy: The accuracy of the explainer model.
        concept_activation_per_image: (n_concepts) The sorted curve of average concept activation per image (see how many concepts are used per image).
        concept_activation_per_concept: (n_concepts) The sorted average activation of each concept per image (see dead concepts).
        raw_concept_mean_activation (n_concepts) The mean activation of each concept over the whole dataset, in the order of concept_correlations.
        concept_correlations: (n_concepts) The correlation of each concept with each other.
    """

    classifier_accuracy: float
    explainer_accuracy: float
    concept_activation_per_image: np.ndarray
    concept_activation_per_concept: np.ndarray
    raw_concept_mean_activation: np.ndarray
    concept_correlations: np.ndarray


def test_concepts(
    # Models
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    # Loader
    dataloader: DataLoader,
    device: str,
) -> ConceptTestResults:
    """Test viscoin concept repartition.

    Args:
        classifier: The classifier model.
        concept_extractor: The concept extractor model.
        explainer: The explainer model.
        dataloader: The testing DataLoader.
        device: The device to use.

    Returns:
        ConceptTestResults: The results of the concept test.
    """

    # Put the models in evaluation mode
    classifier.eval()
    concept_extractor.eval()
    explainer.eval()

    n_concepts = concept_extractor.n_concepts

    # For each image, add the sorted probabilities of the concept to have an idea of how many concepts are activated at once
    concept_activation_per_image = np.zeros(n_concepts)
    # For each concept, measure the activation intensity (via probability), averaged over all images to compute 'dead concepts'
    concept_activation_per_concept = np.zeros(n_concepts)
    # Correlated activation of concepts in a heatmap
    concept_correlations = np.zeros((n_concepts, n_concepts))

    # Compare accuracies
    classifier_accuracies: list[float] = []
    explainer_accuracies: list[float] = []

    for images, labels in tqdm(dataloader, desc="Concept test batches"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            classes, latent = classifier.forward(images)
            encoded_concepts, _ = concept_extractor.forward(latent[-3:])
            explainer_classes = explainer.forward(encoded_concepts)

        preds = classes.argmax(dim=1, keepdim=True)
        preds_expl = explainer_classes.argmax(dim=1, keepdim=True)

        # Compute the accuracy
        classifier_accuracy = preds.eq(labels.view_as(preds)).sum().item() / len(labels)
        explainer_accuracy = preds_expl.eq(labels.view_as(preds_expl)).sum().item() / len(labels)
        classifier_accuracies.append(classifier_accuracy)
        explainer_accuracies.append(explainer_accuracy)

        # Compute concept activations and correlations
        for image_concepts in encoded_concepts:
            # (n_concepts)
            activations = F.adaptive_max_pool2d(image_concepts, 1).squeeze().cpu().numpy()

            concept_activation_per_image += np.sort(activations)
            concept_activation_per_concept += activations
            concept_correlations += np.outer(activations, activations)

    # Divide everything by the amount of images in the dataset
    n_images = len(dataloader.dataset)  # type: ignore

    return ConceptTestResults(
        classifier_accuracy=float(np.mean(classifier_accuracies)),
        explainer_accuracy=float(np.mean(explainer_accuracies)),
        concept_activation_per_image=concept_activation_per_image / n_images,
        concept_activation_per_concept=np.sort(concept_activation_per_concept / n_images),
        raw_concept_mean_activation=concept_activation_per_concept / n_images,
        concept_correlations=concept_correlations / n_images,
    )
