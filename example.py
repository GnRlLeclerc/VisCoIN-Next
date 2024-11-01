"""Example script to showcase the models in action."""

import torch

from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.training.losses import conciseness_diversity_loss

N_CLASSES = 200
N_CONCEPTS = 256


model = Classifier(resnet="50", output_classes=N_CLASSES, pretrained=True)
concept = ConceptExtractor(n_concepts=N_CONCEPTS)
explainer = Explainer(n_concepts=N_CONCEPTS, n_classes=N_CLASSES, normalized=True)

# CUB dataset
dataset = CUB_200_2011()

# Extract 2 images from the dataset to showcase a batch of 2
image, label = dataset[0]
image2, label2 = dataset[1]
image = image.unsqueeze(0)
image2 = image2.unsqueeze(0)

# Build a batch of 2 images with 3 color channels, in 224x224 resolution (default for ResNets)
batch = torch.cat([image, image2], dim=0)
print("Batch:", batch.shape)

# Classifier forward pass
classes, hidden = model.forward(batch)
print("Classifier classes:", classes.shape)

# Process the last 3 concept layers from the classifier with the concept extractor
concept_space, gan_helper_space = concept.forward(hidden[-3:])
print("Concept space:", concept_space.shape)

# Test loss computation on the concept latent space
conciseness_diversity_loss(concept_space)

# Explainer forward pass
classes = explainer.forward(concept_space)
print("Explainer classes:", classes.shape)
