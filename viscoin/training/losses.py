"""Loss functions definitions."""

import lpips
import torch
from torch import Tensor
from torch.nn import functional as F

from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.gan import GeneratorAdapted


def entropy_loss(v: Tensor) -> Tensor:
    """Entropy loss function for a real-valued batched vector `v`.
    It is defined as the negative log loss of the softmax of `v`.

    The entropy is higher when the vector is balanced.

    Args:
        v: (batch_size, n) Real-valued batched vector.
    """
    p = F.softmax(v, dim=1)  # Probabilities
    return -torch.sum(p * torch.log(p))


def l1_loss(x: Tensor) -> Tensor:
    """L1 loss function (difference between `x` and 0 with L1 norm)."""
    return F.l1_loss(x, torch.zeros(x.shape).to(x.device))


###################################################################################################
#                                  CONCEPT SPACE LOSS FUNCTIONS                                   #
###################################################################################################


def conciseness_diversity_loss(concept_embeddings: Tensor, eta=1.0) -> Tensor:
    """Concept conciseness and diversity loss function. Used in FLINT.
    Not used in VisCoIN because it is too strong, and add an additional `eta` hyperparameter.

    Loss = (
        - entropy(mean_concepts_across_batches)  # Use all concepts through all batches
        + entropy(concepts_per_batch)            # Individual samples use only a few concepts
        + eta * l1_norm(concepts_per_batch)      # Regularization (keep norms low)
    )

    Args:
        concept_embeddings: (batch_size, n_concepts, 3, 3) Concept embeddings from the concept extractor.
        eta: Regularization hyperparameter.
    """
    # (batch_size, n_concepts)
    pooled = F.adaptive_max_pool2d(concept_embeddings, 1).flatten(start_dim=1)

    return (
        -entropy_loss(pooled.mean(dim=0).unsqueeze(0))
        + entropy_loss(pooled)
        + eta * l1_loss(pooled)
    )


def concept_regularization_loss(concept_embeddings: Tensor) -> Tensor:
    """Concept regularization loss function. Used in VisCoIN.

    Loss = (
        l1_norm(all_normalized_concepts)  # Encourage sparsity
        + l1_norm(concept_embeddings)     # Regularization (keep norms low)
    )

    Args:
        concept_embeddings: (batch_size, n_concepts, 3, 3) Concept embeddings from the concept extractor.
    """
    # (batch_size, n_concepts)
    pooled = F.adaptive_max_pool2d(concept_embeddings, 1).flatten(start_dim=1)
    normed = F.normalize(pooled, p=2, dim=1)

    return l1_loss(normed) + l1_loss(concept_embeddings)


def concept_orthogonality_loss(model: ConceptExtractor) -> Tensor:
    """Additional concept loss to enforce orthogonality between concepts.

    Args:
        model: The concept extractor model, whose weights will be used to compute the loss.
    """

    # Gather weights from the last convolutional layer before the concept 3x3 embeddings
    # and view it as shape (n_concepts, -1) to lay out vector weights for each concept
    concept_weights = model.conv5.weight.view(model.n_concepts, -1)
    normed_weights = F.normalize(concept_weights, dim=1).abs()

    return (normed_weights @ normed_weights.T - model.n_concepts).sum() / (model.n_concepts**2)


###################################################################################################
#                                  RECONSTRUCTION LOSS FUNCTIONS                                  #
###################################################################################################

# Cache the LPIPS network to avoid loading it every time
_lpips_network = lpips.LPIPS(net="vgg")


def lpips_loss(reconstructed: Tensor, original: Tensor) -> Tensor:
    """LPIPS loss function.

    Args:
        reconstructed: (batch_size, 3, H, W) Reconstructed images.
        original: (batch_size, 3, H, W) Original images.
    """
    return torch.mean(_lpips_network.to(reconstructed.device)(reconstructed, original))


def reconstruction_loss(
    reconstructed: Tensor,
    original: Tensor,
    reconstructed_classes: Tensor,
    original_classes: Tensor,
    lambda_classes=0.1,
    lambda_lpips=3.0,
) -> Tensor:
    """Image reconstruction loss function.

    Loss = (
        l1_norm(reconstructed - original)  # L1 loss
        + l2_norm(reconstructed - original)  # L2 loss
        + lambda_lpips * lpips(reconstructed, original)  # LPIPS loss (perceptual similarity metric)
        + lambda_classes * classification_loss(reconstructed, original)  # f(original) - f(reconstructed) classes comparison
    )

    Args:
        reconstructed: (batch_size, 3, H, W) Reconstructed images.
        original: (batch_size, 3, H, W) Original images.
        reconstructed_classes: (batch_size, n_classes) Reconstructed classes normalized probabilities.
        original_classes: (batch_size, n_classes) Original classes normalized probabilities.
        lambda_classes: Classification loss weight.
        lambda_lpips: LPIPS loss weight.
    """

    return (
        F.l1_loss(reconstructed, original)
        + F.mse_loss(reconstructed, original)
        + lambda_classes * F.cross_entropy(reconstructed_classes, original_classes.detach())
        + lambda_lpips * lpips_loss(reconstructed, original)
    )


###################################################################################################
#                                  OUTPUT FIDELITY LOSS FUNCTIONS                                 #
###################################################################################################


def output_fidelity_loss(original_classes: Tensor, explainer_classes: Tensor) -> Tensor:
    """Output fidelity loss function. Compares the predictions of the original classifier and
    those of the explainer network.

    Args:
        original_classes: (batch_size, n_classes) Original classes unnormalized logits.
        explainer_classes: (batch_size, n_classes) Explainer classes unnormalized logits.
    """

    return F.cross_entropy(explainer_classes, original_classes.detach())


###################################################################################################
#                                      STYLEGAN LOSS FUNCTION                                     #
###################################################################################################


def gan_regularization_loss(gan_latents: Tensor, model: GeneratorAdapted) -> Tensor:
    """StyleGAN regularization loss function.

    Args:
        gan_latents: (batch_size, w_dim) StyleGAN latents (ws).
        mode: GeneratorAdapted StyleGAN model.
    """

    w_mapping = model.mapping.fixed_w_avg.repeat([gan_latents.shape[0], gan_latents.shape[1], 1])

    return F.mse_loss(gan_latents, w_mapping.detach())
