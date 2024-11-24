"""Example script to showcase the models in action."""

import matplotlib.pyplot as plt
import torch

from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.training.losses import concept_regularization_loss
from viscoin.utils.images import from_torch

N_CLASSES = 200
N_CONCEPTS = 256

DEVICE = "cuda"


model = Classifier(output_classes=N_CLASSES, pretrained=True).to(DEVICE)
concept = ConceptExtractor(n_concepts=N_CONCEPTS).to(DEVICE)
explainer = Explainer(n_concepts=N_CONCEPTS, n_classes=N_CLASSES, normalized=True).to(DEVICE)

# GAN model
mapping_net_params = {"num_layers": 1, "fixed_w_avg": None, "coarse_layer": 2, "mid_layer": 10}
generator = GeneratorAdapted(
    z_dim=N_CONCEPTS,
    c_dim=0,
    w_dim=512,
    img_resolution=256,
    img_channels=3,
    mapping_kwargs=mapping_net_params,
).to(DEVICE)

# CUB dataset
dataset = CUB_200_2011()

# Extract 2 images from the dataset to showcase a batch of 2
image, label = dataset[0]
image2, label2 = dataset[1]
image = image.unsqueeze(0)
image2 = image2.unsqueeze(0)

# Build a batch of 2 images with 3 color channels, in 224x224 resolution (default for ResNets)
batch = torch.cat([image, image2], dim=0).to(DEVICE)
print("Batch:", batch.shape)

# Classifier forward pass
classes, hidden = model.forward(batch)
print("Classifier classes:", classes.shape)

# Process the last 3 concept layers from the classifier with the concept extractor
concept_space, gan_helper_space = concept.forward(hidden[-3:])
print("Concept space:", concept_space.shape)

# Test loss computation on the concept latent space
concept_regularization_loss(concept_space)

# Explainer forward pass
classes = explainer.forward(concept_space)
print("Explainer classes:", classes.shape)

# Generate images with the GAN model
fake_img, ws = generator(z1=concept_space, z2=gan_helper_space, return_latents=True)
print("Fake image:", fake_img.shape)

images = from_torch(fake_img)
print("Images:", images.shape)

plt.imshow(images[0])
plt.show()

plt.imshow(images[1])
plt.show()
