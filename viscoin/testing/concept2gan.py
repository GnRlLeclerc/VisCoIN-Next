import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from viscoin.models.classifiers import Classifier
from viscoin.models.concept2gan import Concept2GAN
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.utils.metrics import logits_accuracy


def test_concept2gan(
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    concept2gan: Concept2GAN,
    explainer: Explainer,
    dataloader: DataLoader,
    device: str,
    verbose=True,
):
    """Test the Concept2GAN pipeline on a dataloader"""

    classifier.eval()
    concept_extractor.eval()
    concept2gan.eval()
    explainer.eval()

    accuracy = 0
    w_loss = 0

    with torch.no_grad():
        for images, _, w_plus in tqdm(
            dataloader, desc="Testing Concept2GAN pipeline", disable=not verbose
        ):
            images = images.to(device)
            w_plus = w_plus.to(device)

            # Get predictions and latents from the classifier
            preds, latents = classifier.forward(images)
            # Extract concepts from the latents
            concepts, _ = concept_extractor.forward(latents[-3:])
            # Rebuild predictions from concepts
            concept_preds = explainer.forward(concepts)
            # Rebuild W+ code from concepts
            w_plus_rebuilt = concept2gan.forward(concepts)

            accuracy += logits_accuracy(concept_preds, preds)
            w_loss += F.mse_loss(w_plus_rebuilt, w_plus.to(device)).item()

    return {
        "test_accuracy": accuracy / len(dataloader),
        "test_w_loss": w_loss / len(dataloader),
    }
