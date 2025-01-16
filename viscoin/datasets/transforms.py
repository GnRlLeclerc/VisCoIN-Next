"""Dataset image transformations.

Taken from the Pytorch Resnet page: https://pytorch.org/hub/pytorch_vision_resnet/
"""

import torch
from torchvision.transforms import v2 as transforms

"""
- RandomResizedCrop: focus on random aspects of the image
- RandomHorizontalFlip: double the dataset size by flipping some images horizontally
- Normalize: normalize the image to the pretrained ImageNet mean and standard deviation
"""
RESNET_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


"""
- Resize + CenterCrop: eliminate the outer part of the image to remove background noise
- Normalize: normalize the image to the pretrained ImageNet mean and standard deviation
"""
RESNET_TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(256 / 0.875)),
        transforms.CenterCrop(256),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
