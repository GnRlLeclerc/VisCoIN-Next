"""Utility types"""

from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from torch.types import Number

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

    @staticmethod
    def plot_losses(results: list["TestingResults"]):

        acc_loss = [result.acc_loss for result in results]
        cr_loss = [result.cr_loss for result in results]
        of_loss = [result.of_loss for result in results]
        lp_loss = [result.lp_loss for result in results]
        rec_loss_l1 = [result.rec_loss_l1 for result in results]
        rec_loss_l2 = [result.rec_loss_l2 for result in results]

        # Plot the losses
        plt.plot(acc_loss, label="acc_loss")
        plt.plot(cr_loss, label="cr_loss")
        plt.plot(of_loss, label="of_loss")
        plt.plot(lp_loss, label="lp_loss")
        plt.plot(rec_loss_l1, label="rec_loss_l1")
        plt.plot(rec_loss_l2, label="rec_loss_l2")
        plt.title("Testing Losses")
        plt.ylabel("Loss")
        plt.xlabel("per 20,000 batches")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_preds_overlap(results: list["TestingResults"]):

        preds_overlap = [result.preds_overlap for result in results]
        correct_preds = [result.correct_preds for result in results]
        correct_expl_preds = [result.correct_expl_preds for result in results]

        # Plot the prediction overlaps
        plt.plot(preds_overlap, label="preds_overlap")
        plt.plot(correct_preds, label="correct_preds")
        plt.plot(correct_expl_preds, label="correct_expl_preds")
        plt.title("Prediction Overlaps")
        plt.ylabel("Overlap")
        plt.xlabel("per 20,000 batches")
        plt.legend()
        plt.show()


@dataclass
class TrainingResults:
    """Training results for the VisCoIN ensemble.

    Args:
        acc_loss: The accuracy loss.
        cr_loss: The concept regularization loss.
        of_loss: The output fidelity loss.
        ortho_loss: The concept orthogonality loss.
        rec_loss: The reconstruction loss.
        gan_loss: The GAN regularization loss.
        inter_loss: The intersection loss.
    """

    acc_loss: Number
    cr_loss: Number
    of_loss: Number
    ortho_loss: Number
    rec_loss: Number
    gan_loss: Number
    inter_loss: Number

    @staticmethod
    def plot_losses(results: list["TrainingResults"]):

        acc_loss = [result.acc_loss for result in results]
        cr_loss = [result.cr_loss for result in results]
        of_loss = [result.of_loss for result in results]
        ortho_loss = [result.ortho_loss for result in results]
        rec_loss = [result.rec_loss for result in results]
        gan_loss = [result.gan_loss for result in results]
        inter_loss = [result.inter_loss for result in results]

        # Plot the losses
        plt.plot(acc_loss, label="acc_loss")
        plt.plot(cr_loss, label="cr_loss")
        plt.plot(of_loss, label="of_loss")
        plt.plot(ortho_loss, label="ortho_loss")
        plt.plot(rec_loss, label="rec_loss")
        plt.plot(gan_loss, label="gan_loss")
        plt.plot(inter_loss, label="inter_loss")
        plt.title("Training Losses")
        plt.ylabel("Loss")
        plt.xlabel("per 20,000 batches")
        plt.legend()
        plt.show()
