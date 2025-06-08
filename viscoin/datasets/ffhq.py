import os

import kagglehub
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose as ComposeV1,  # for the CLIP model that uses legacy compose
)
from torchvision.transforms.v2 import Compose as ComposeV2

from viscoin.datasets.transforms import RESNET_TEST_TRANSFORM, RESNET_TRAIN_TRANSFORM
from viscoin.utils.types import Mode

Compose = ComposeV1 | ComposeV2


class FFHQDataset(Dataset):
    """FFHQ dataset with features annotations from https://github.com/DCGM/ffhq-features-dataset.

    The dataset contains images of shape 256x256 pixels."""

    def __init__(self, mode: Mode = "train", transform: Compose | None = None) -> None:

        self.dataset_path = kagglehub.dataset_download("denislukovnikov/ffhq256-images-only")
        self.dataset_path = os.path.join(self.dataset_path, "ffhq256")
        self.image_cache = {}

        self.mode = mode

        # Load appropriate transformations if none are provided
        if transform is None:
            if self.mode == "train":
                transform = RESNET_TRAIN_TRANSFORM
            else:
                transform = RESNET_TEST_TRANSFORM
        self.transform = transform

    def load_image(self, index: int) -> Tensor:
        """Loads an image from the dataset at the given index."""
        image_path = os.path.join(self.dataset_path, f"{index:05d}.png")
        image = self.transform(Image.open(image_path).convert("RGB"))

        return image  # type: ignore

    def __len__(self) -> int:
        return 70_000

    def __getitem__(self, index: int) -> Tensor:

        if index in self.image_cache:
            image = self.image_cache[index]
        else:
            image = self.load_image(index)
            self.image_cache[index] = image

        return image
