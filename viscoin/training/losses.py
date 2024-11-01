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
