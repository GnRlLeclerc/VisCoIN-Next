# VisCoIN

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/pytorch-v2.5.0-orange?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Version" />
  <img src="https://img.shields.io/badge/code%20formatter-black-000000?style=for-the-badge&logo=python&logoColor=white" alt="Black Formatter" />
</div>

<br>

Implementation of Vision Concept-based Interpretable Networks. [See the paper](https://arxiv.org/abs/2407.01331v1).

This project uses the [Black](https://github.com/psf/black) python formatter. Imports are sorted using [isort](https://pycqa.github.io/isort/).

See also (used in this repository):

- [StyleGAN2 ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [CLIP](https://github.com/openai/CLIP)

## Project Structure

Structure of the `viscoin` folder.

```bash
├── ../stylegan2_ada  # Pytorch implementation of StyleGAN2 ADA
│
├── datasets       # Pytorch dataloaders for every dataset
│   └── cub                 # CUB dataset loader
│
├── models         # Model definitions
│   ├── classifiers         # Classifier model (`f`)
│   ├── concept_extractors  # From classifier latent to concepts (`Psi` in VisCoIN)
│   └── explainers          # From concepts to class (`Theta` in VisCoIN)
│
├── testing        # Testing functions
│
└── training       # Training functions
    └── losses              # Loss functions
```

## Quickstart

Create a conda environment for the project with these commands:

```bash
conda env create -f conda.yml  # Create the `viscoin` environment
conda activate viscoin         # Activate it
```

Clone the StyleGAN2 ADA submodule:

```bash
git submodule update --init
```

The StyleGAN uses a custom CUDA plugin. You need to install a CUDA compiler (`nvcc`), and export the `CUDA_HOME` environment variable.

Last, the example script:

```bash
python example.py
```
