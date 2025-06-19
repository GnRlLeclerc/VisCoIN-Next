"""VisCoIN metrics"""

import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_matching(original: Tensor, rebuilt: Tensor) -> float:
    """Given 2 tensors of embeddings, compute the proportion of rebuilt embeddings
    that match best with the original embeddings in the same index position,
    according to cosine similarity.

    Args:
        original (n, embed_size): The original embeddings
        rebuilt (n, embed_size): The rebuilt embeddings

    Returns:
        match ratio in [0, 1]
    """

    assert original.shape == rebuilt.shape, "Tensors must have the same shape"
    assert original.dim() == 2, "Tensors must be 2D"

    similarities = F.cosine_similarity(original[:, None, :], rebuilt, dim=2)

    # Highest similarity values for each row
    highest = torch.max(similarities, dim=1)

    # Similarity value of the original pairs
    arange = torch.arange(original.shape[0])
    diagonal = similarities[arange, arange]

    # NOTE: we compare by value and not index, as multiple embeddings can have the same similarity
    correct = torch.sum(highest.values == diagonal)

    return correct.item() / original.shape[0]


def logits_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute the accuracy of predicted logits against targets logits.
    Useful to evaluate output fidelity of an explainer vs a classifier.

    Args:
        logits (batch_size, num_classes): Predicted logits
        targets (batch_size, num_classes): Ground truth logits

    Returns:
        accuracy in [0, 1]
    """

    assert logits.shape == targets.shape, "Logits and targets must have the same shape"
    assert logits.dim() == 2, "Logits and targets must be 2D"

    # Get the predicted class indices
    predicted_classes = logits.argmax(dim=1)
    target_classes = targets.argmax(dim=1)

    # Compute the accuracy
    correct_predictions = (predicted_classes == target_classes).sum().item()
    accuracy = correct_predictions / logits.size(0)

    return accuracy
