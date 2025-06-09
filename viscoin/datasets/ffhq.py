import json
import os
from typing import Literal

import kagglehub
import numpy as np
from PIL import Image
import torch
from torch import Tensor, tensor
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose as ComposeV1,  # for the CLIP model that uses legacy compose
)
from torchvision.transforms.v2 import Compose as ComposeV2

from viscoin.datasets.transforms import RESNET_TEST_TRANSFORM, RESNET_TRAIN_TRANSFORM
from viscoin.utils.types import Mode

Compose = ComposeV1 | ComposeV2

AttrFFHQ = Literal["smile", "gender", "age", "glasses", "emotion", "makeup", "hair"]


class FFHQDataset(Dataset):
    """FFHQ dataset with features annotations from https://github.com/DCGM/ffhq-features-dataset.

    The dataset contains images of shape 256x256 pixels."""

    def __init__(
        self, mode: Mode = "train", transform: Compose | None = None, attr: AttrFFHQ = "emotion"
    ) -> None:
        """Instanciate a FFHQ dataset.

        Args:
            mode: Whether to consider training or testing data. Defaults to "train".
            transform: Test, train or custom transforms for images.
            attr: The attribute to use for the labels. Defaults to gender.
        """
        self.dataset_path = kagglehub.dataset_download("denislukovnikov/ffhq256-images-only")
        self.dataset_path = os.path.join(self.dataset_path, "ffhq256")
        self.image_cache = {}
        self.label_cache = {}
        self.attr = attr

        self.mode = mode

        # Load appropriate transformations if none are provided
        if transform is None:
            if self.mode == "train":
                transform = RESNET_TRAIN_TRANSFORM
            else:
                transform = RESNET_TEST_TRANSFORM
        self.transform = transform

        # Load train test split
        indexes = np.loadtxt("viscoin/datasets/ffhq_split.txt", dtype=int, delimiter=" ")
        self.train_indexes = indexes[indexes[:, 1] == 1][:, 0]
        self.test_indexes = indexes[indexes[:, 1] == 0][:, 0]

    def load_image(self, index: int) -> tuple[Tensor, Tensor]:
        """Loads an image from the dataset at the given index."""
        image_path = os.path.join(self.dataset_path, f"{index:05d}.png")
        image = self.transform(Image.open(image_path).convert("RGB"))
        self.image_cache[index] = image

        label_path = os.path.join("ffhq-annotations", "json", f"{index:05d}.json")
        with open(label_path, "r") as f:
            json_labels = json.load(f)
            if len(json_labels) == 0:
                label = _default(self.attr)
            else:
                label = _extract_attr(json_labels[0], self.attr)
            self.label_cache[index] = label

        return image, label  # type: ignore

    def __len__(self) -> int:
        """Returns the length of the dataset (depends on the test/train mode)."""

        if self.mode == "train":
            return len(self.train_indexes)
        return len(self.test_indexes)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:

        # Get the absolute index
        index = self.train_indexes[index] if self.mode == "train" else self.test_indexes[index]

        if index in self.image_cache:
            image = self.image_cache[index]
            label = self.label_cache[index]
        else:
            image, label = self.load_image(index)

        return image, label


def _extract_attr(label: dict, attr: AttrFFHQ) -> Tensor:
    """Extract the attribute of interest from the label dictionary."""

    fa = label["faceAttributes"]
    emo = fa["emotion"]

    match attr:
        case "age":
            return tensor(fa["age"])
        case "gender":
            return tensor(1.0 if fa["gender"] == "male" else 0.0)
        case "glasses":
            return tensor(1.0 if fa["glasses"] != "NoGlasses" else 0.0)
        case "hair":
            return tensor(1 - fa["hair"]["bald"])
        case "smile":
            return tensor(fa["smile"])
        case "makeup":
            # 2 values: eye & lip
            return tensor(
                [
                    float(fa["makeup"]["eyeMakeup"]),
                    float(fa["makeup"]["lipMakeup"]),
                ]
            )
        case "emotion":
            # One-hot encoding of the emotion
            return tensor(
                [
                    float(emo["anger"]),
                    float(emo["contempt"]),
                    float(emo["disgust"]),
                    float(emo["fear"]),
                    float(emo["happiness"]),
                    float(emo["neutral"]),
                    float(emo["sadness"]),
                    float(emo["surprise"]),
                ]
            )


def _default(attr: AttrFFHQ) -> Tensor:
    """Default value for the attribute if it is missing from the dataset."""

    match attr:
        case "age" | "gender" | "glasses" | "hair" | "smile":
            return tensor(0.0)
        case "makeup":
            return torch.zeros(2)
        case "emotion":
            return torch.zeros(8)
