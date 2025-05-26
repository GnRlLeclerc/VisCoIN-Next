import click
import torch
import numpy as np

from viscoin.models.clip import CLIP

from viscoin.cli.utils import (
    dataset,
    device,
    viscoin_pickle_path,
    concept2clip_pickle_path,
    clip_version,
)

from viscoin.datasets.cub import CUB_200_2011
from viscoin.datasets.funnybirds import FunnyBirds

from viscoin.models.utils import load_viscoin_pickle
from viscoin.testing.concept2clip import get_concept_labels_vocab

from viscoin.captions.cub import load as load_cub_vocab


@click.command()
@concept2clip_pickle_path
@viscoin_pickle_path
@click.option(
    "--n-concepts",
    help="The number of concepts",
    type=int,
    default=256,
)
@dataset
@click.option(
    "--amplify-multiplier",
    help="The multiplier to amplify to the concept",
    type=float,
    default=4.0,
)
@click.option(
    "--selection-n",
    help="The number of images to select for each concept",
    type=int,
    default=100,
)
@click.option(
    "--output_path",
    help="The path to save the concept labels",
    type=str,
    default="concept_labels.csv",
)
@clip_version
@device
def clip_concept_labels(
    concept2clip_pickle_path: str,
    viscoin_pickle_path: str,
    n_concepts: int,
    dataset: str,
    amplify_multiplier: float,
    selection_n: int,
    output_path: str,
    clip_version: str,
    device: str,
):
    """
    Generate concept labels from a given vocabulary using the clip adapter model :

        We select the most activating images for each concept based on a threshold.
        For each image, we compute its CLIP embedding and the CLIP embedding of the image where the concept is amplified.
        We compute the difference between the two and compute the similarity with the vocabulary.
        We average the similarity for each concept and save the results to a file.

    Args:
        clip_adapter_path (str): Path to the clip adapter model
        vocab_path (str): Path to the vocabulary file (.txt)
        viscoin_pickle_path (str): Path to the viscoin pickle file
        n_concepts (int): The number of concepts
        dataset_path (str): Path to the dataset
        amplify_multiplier (float): The multiplier to amplify to the concept
        selection_n (int): The number of images to select for each concept
        output_path (str): The path to save the concept labels
        clip_version (str): The version of the CLIP model to use
    """

    # Load CLIP adapter model and CLIP model
    concept2clip = torch.load(concept2clip_pickle_path, weights_only=False).to(device)

    clip_model = CLIP(clip_version)

    # Load Viscoin Classifier and Concept Extractor
    viscoin = load_viscoin_pickle(viscoin_pickle_path)
    viscoin.classifier = viscoin.classifier.to(device)
    viscoin.concept_extractor = viscoin.concept_extractor.to(device)

    # Load the dataset
    match dataset:
        case "cub":
            dataset = CUB_200_2011()
            vocab = load_cub_vocab()
        case _:
            raise ValueError(f"Dataset type {dataset} not supported")

    concept_labels, probs, most_activating_images = get_concept_labels_vocab(
        concept2clip,
        viscoin.concept_extractor,
        viscoin.classifier,
        clip_model,
        vocab,
        n_concepts,
        dataset,
        amplify_multiplier,
        selection_n,
        device,
    )

    # Save to file
    with open(output_path, "w") as f:

        f.write("unit,description,similarity,most-activating-images\n")

        for i, label in enumerate(concept_labels):
            f.write(
                f"{i},{label},{probs[i]},{":".join(np.char.mod("%i", most_activating_images[i]))}\n"
            )
