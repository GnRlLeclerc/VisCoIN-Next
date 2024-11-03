"""CUB 200 2011 dataset loader.

https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

Be sure to put the "attributes.txt" file in the "CUB_200_2011" folder.

The dataset contains images of birds, class labels, bounding boxes and parts annotations.
We do not load the parts annotations.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from viscoin.utils.types import Mode


class CUB_200_2011(Dataset):
    """CUB 200 2011 dataset loader."""

    def __init__(
        self,
        dataset_path="datasets/CUB_200_2011/",
        mode: Mode = "train",
        image_shape: tuple[int, int] = (224, 224),
        bbox_only=False,
        transform: Compose = Compose([ToTensor()]),
    ) -> None:
        """Instantiate a CUB dataset. Its result is saved in a pickle file for faster reloading.

        Args:
            dataset_path: Path to the downloaded dataset. Defaults to "datasets/CUB_200_2011".
            mode: Whether to consider training or testing data. Defaults to "train".
            image_shape: the shape to resize each image (the dataset does not have normalized shapes). Note that (224, 224) is the default shape for ResNets.
            bbox_only: Whether to crop the images to include only the bounding box of the bird.
            transform: Additional optional transformations to perform on loaded images.
        """

        assert os.path.exists(dataset_path), f'Dataset path "{dataset_path}" not found.'

        self.dataset_path = dataset_path
        self.mode: Mode = mode
        self.image_shape = image_shape
        self.bbox_only = bbox_only
        self.transform = transform

        # Load the metadata
        # Extract training and testing image indexes
        indexes = np.loadtxt(f"{self.dataset_path}train_test_split.txt", dtype=int, delimiter=" ")
        self.train_indexes = indexes[indexes[:, 1] == 1][:, 0] - 1
        self.test_indexes = indexes[indexes[:, 1] == 0][:, 0] - 1

        # Read labels
        labels = np.loadtxt(f"{self.dataset_path}image_class_labels.txt", dtype=int, delimiter=" ")
        labels[:, 1] -= 1  # Labels start at 1 in the file
        self.labels = labels[:, 1]  # Remove the image index

        # Read image paths
        image_paths = np.loadtxt(f"{self.dataset_path}images.txt", dtype=str, delimiter=" ")
        self.image_paths = image_paths[:, 1]  # Remove the image index

        # Read bounding boxes
        bboxes = np.loadtxt(f"{self.dataset_path}bounding_boxes.txt", dtype=int, delimiter=" ")
        self.bboxes = bboxes[:, 1:]  # Remove the image index

        # Image cache
        # The whole dataset is not loaded instantly in memory, but only when needed.
        self.image_cache: dict[int, Tensor] = {}

    def load_image(self, index: int) -> Tensor:
        """Load an image by index, and apply the specified transformations.
        Note that tensors have reversed dimensions (C, H, W) instead of (H, W, C)."""

        # Load the image
        image = Image.open(f"{self.dataset_path}images/{self.image_paths[index]}")

        # Convert to RGB if needed
        if image.getbands() == ("L",):
            image = image.convert("RGB")

        if self.bbox_only:
            # Crop the image to include only the bounding box
            x, y, width, height = self.bboxes[index]
            image = image.resize(self.image_shape, box=(x, y, x + width, y + height))
        else:
            image = image.resize(self.image_shape)

        # Apply the transformations
        tensor_image = self.transform(image)

        if type(tensor_image) is not Tensor:
            raise ValueError("The transform must return a tensor.")

        return tensor_image

    def set_mode(self, mode: Mode):
        """Set the mode of the dataset (train or test).
        Use this instead of creating 2 different instances of this dataset."""

        self.mode = mode

    def __len__(self):
        """Returns the length of the dataset (depends on the test/train mode)."""

        if self.mode == "train":
            return len(self.train_indexes)
        return len(self.test_indexes)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get an image and its label by index. The index is over training or testing data.

        Returns:
            image: The image tensor.
            label: The label tensor.
        """
        # Get the absolute index
        index = self.train_indexes[index] if self.mode == "train" else self.test_indexes[index]

        # Load the image
        if index in self.image_cache:
            image = self.image_cache[index]
        else:
            image = self.load_image(index)
            self.image_cache[index] = image

        label = torch.as_tensor(self.labels[index])

        return image, label
