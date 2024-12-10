"""Utility types"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# Training mode vs testing mode
Mode = Literal["train", "test"]

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
