"""Loss functions definitions."""

import torch
from torch import Tensor
from torch.nn import functional as F


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


def conciseness_diversity_loss(concept_embeddings: Tensor, eta=1.0) -> Tensor:
    """Concept conciseness and diversity loss function.

    Loss = (
        - entropy(mean_concepts_across_batches)  # Use all concepts through all batches
        + entropy(concepts_per_batch)            # Individual samples use only few concepts
        + eta * l1_norm(concepts_per_batch)      # Regularization (keep norms low)
    )

    Args:
        concept_embeddings: (batch_size, n_concepts, 3, 3) Concept embeddings from the concept extractor.
        eta: Regularization hyperparameter.
    """
    n_concepts = concept_embeddings.shape[1]

    # (batch_size, n_concepts)
    pooled = F.adaptive_max_pool2d(concept_embeddings, 1).view(-1, n_concepts)

    return (
        -entropy_loss(pooled.mean(dim=0).unsqueeze(0))
        + entropy_loss(pooled)
        + eta * l1_loss(pooled)
    )
