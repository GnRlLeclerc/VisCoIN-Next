"""Testing functions for the viscoin ensemble."""

# pyright: reportPossiblyUnboundVariable=false

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.types import Number
from torch.utils.data import DataLoader
from tqdm import tqdm

from stylegan2_ada.dnnlib.util import open_url
from stylegan2_ada.metrics.metric_utils import FeatureStats
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.training.losses import (
    concept_regularization_loss,
    lpips_loss,
    output_fidelity_loss,
)

Float = np.floating[Any]


@dataclass
class TestingResults:
    """VisCoIN testing results.

    Args:
        acc_loss: The accuracy loss.
        cr_loss: The concept regularization loss.
        of_loss: The output fidelity loss.
        lp_loss: The LPIPS loss.
        rec_loss_l1: The L1 reconstruction loss.
        rec_loss_l2: The L2 reconstruction loss.
        preds_overlap: The overlap between the classifier and explainer predictions as a percentage.
        correct_preds: The percentage of correct classifier predictions.
        correct_expl_preds: The percentage of correct explainer predictions.
        fid_score: The Fréchet Inception Distance score if computed.
    """

    acc_loss: Float
    cr_loss: Float
    of_loss: Float
    lp_loss: Float
    rec_loss_l1: Float
    rec_loss_l2: Float
    preds_overlap: Float
    correct_preds: Float
    correct_expl_preds: Float
    fid_score: Float | None = None


def test_viscoin(
    # Models
    classifier: Classifier,
    concept_extractor: ConceptExtractor,
    explainer: Explainer,
    viscoin_gan: GeneratorAdapted,
    # Loader
    dataloader: DataLoader,
    device: str,
    # FID loss
    compute_fid: bool = False,
    verbose: bool = True,
) -> TestingResults:
    """Test the classifier performance across a testing Dataloader

    Args:
        classifier: The classifier model.
        concept_extractor: The concept extractor model.
        explainer: The explainer model.
        viscoin_gan: The VisCoIN GAN model.
        dataloader: The testing DataLoader.
        device: The device to use.
        compute_fid: Whether to compute the Fréchet Inception Distance score.
        verbose: Whether to display the progress bar.

    Returns:
        The testing results in a dataclass.
    """

    # Put the models in evaluation mode
    classifier.eval()
    concept_extractor.eval()
    explainer.eval()
    viscoin_gan.eval()

    # Create the loss arrays
    acc_loss: list[Number] = []
    cr_loss: list[Number] = []
    of_loss: list[Number] = []
    lp_loss: list[Number] = []
    rec_loss_l1: list[Number] = []
    rec_loss_l2: list[Number] = []
    preds_overlap: list[Number] = []
    correct_preds: list[Number] = []
    correct_expl_preds: list[Number] = []

    # Prepare the feature detector model
    if compute_fid:
        detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
        detector_kwargs = {"return_features": True}
        with open_url(detector_url, verbose=True) as f_detector:
            feature_detector = torch.jit.load(f_detector).eval().to(device)
        stats_real = FeatureStats(max_items=len(dataloader.dataset), capture_mean_cov=True)  # type: ignore
        stats_fake = FeatureStats(max_items=len(dataloader.dataset), capture_mean_cov=True)  # type: ignore

    for images, labels in tqdm(dataloader, desc="Viscoin test batches", disable=not verbose):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            classes, latent = classifier.forward(images)
            encoded_concepts, extra_info = concept_extractor.forward(latent[-3:])
            explainer_classes = explainer.forward(encoded_concepts)
            rebuilt_images: Tensor = viscoin_gan.forward(z1=encoded_concepts, z2=extra_info)  # type: ignore
        preds = classes.argmax(dim=1, keepdim=True)
        preds_expl = explainer_classes.argmax(dim=1, keepdim=True)

        # Compute the different losses
        acc_loss.append(F.cross_entropy(classes, labels).item())
        cr_loss.append(concept_regularization_loss(encoded_concepts).item())
        of_loss.append(output_fidelity_loss(classes, explainer_classes).item())
        lp_loss.append(lpips_loss(rebuilt_images, images).item())
        rec_loss_l1.append(F.l1_loss(rebuilt_images, images).item())
        rec_loss_l2.append(F.mse_loss(rebuilt_images, images).item())
        preds_overlap.append(torch.sum(preds == preds_expl).item())
        correct_preds.append(torch.sum(preds == labels).item())
        correct_expl_preds.append(torch.sum(preds_expl == labels).item())

        if compute_fid:
            fake_features = feature_detector(rebuilt_images, **detector_kwargs)
            real_features = feature_detector(images, **detector_kwargs)
            stats_fake.append_torch(fake_features)
            stats_real.append_torch(real_features)

    # Aggregate the different losses
    results = TestingResults(
        acc_loss=np.mean(acc_loss),
        cr_loss=np.mean(cr_loss),
        of_loss=np.mean(of_loss),
        lp_loss=np.mean(lp_loss),
        rec_loss_l1=np.mean(rec_loss_l1),
        rec_loss_l2=np.mean(rec_loss_l2),
        preds_overlap=100 * np.mean(preds_overlap),
        correct_preds=100 * np.mean(correct_preds),
        correct_expl_preds=100 * np.mean(correct_expl_preds),
    )

    if compute_fid:
        mu_real, sigma_real = stats_real.get_mean_cov()
        mu_fake, sigma_fake = stats_fake.get_mean_cov()
        m = np.square(mu_fake - mu_real).sum()
        s, _ = sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_fake + sigma_real - s * 2))
        results.fid_score = fid

    return results
