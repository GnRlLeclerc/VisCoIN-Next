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
│   ├── cub                 # CUB dataset loader
│   └── transforms          # Standard dataset preprocessing transformations
│
├── models         # Model definitions
│   ├── classifiers         # Classifier model (`f`)
│   ├── concept_extractors  # From classifier latent to concepts (`Psi` in VisCoIN)
│   ├── explainers          # From concepts to class (`Theta` in VisCoIN)
│   └── gan                 # StyleGAN implementation (modified stylegan2_ada)
│
├── testing        # Testing functions
│   └── classifiers         # Testing function for the classifier
│
└── training       # Training functions
    ├── classifiers         # Training function for the classifier
    └── losses              # Loss functions
```

## Quickstart

Create a conda environment for the project with these commands:

```bash
conda env create -f conda.yml  # Create the `viscoin` environment
conda activate viscoin         # Activate it
```

If you encounter an issue, you might have to run the following commands, depending on your conda implementation:

```bash
conda config --set channel_priority flexible
micromamba config set channel_priority flexible
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

## Running jobs with SLURM

There are 2 ways to run jobs on a server that allocates GPUs using SLURM.

Run `sinfo` to see a list of available partitions.

### Interactive

You can run scripts interactively in the terminal with `srun`:

```bash
srun --gpus=1 --partition=P100 --pty bash  # Open a shell on a GPU node
srun --gpus=1 --partition=P100 nvidia-smi  # Display the GPU capabilities of a node

# Run a full job interactively
srun --gpus=1 --partition=P100 --time=00:10:00 --nodes=1 --gpus=1 python main.py test resnet50 --dataset-path datasets/CUB_200_2011/ --batch-size 512
```

### Background

In order to run jobs in the background, you have to define bash scripts. Jobs running in the background will write their output to a `slurm-<job-id>.out` file.

Example script:

```bash
/usr/bin/env bash

# Partition for the job:
#SBATCH --partition=P100

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# Maximum number of GPUS used by the job:
#SBATCH --gpus=1

# The maximum running time of the job in days-hours:mins:sec (here: 1h)
#SBATCH --time=0-01:00:00

srun python main.py test resnet50 --dataset-path datasets/CUB_200_2011/ --batch-size 512
```

You then have to run the script (and interact with it) using these commands:

```bash
sbatch my-script.bash  # Launch the background job
squeue -j <job-id>  # See whether the job it still running
```

You can monitor the output using the following command:

```bash
# You could also use `watch -n 10`, but it has an issue with progress bars
while true; do clear && cat slurm-<job-id>.out; sleep 10; done
```

## Hyperparameters

Hyperparameters used on different datasets for the training of the different VisCoIN models.

### Classifier

| Dataset       | Optimizer | Learning Rate              | Epochs |
| ------------- | --------- | -------------------------- | ------ |
| CUB           | Adam      | 0.0001                     | 30     |
| CelebA-HQ     | Adam      | 0.001                      | 10     |
| Stanford-Cars | SGD       | 0.1 (x0.1 every 30 epochs) | 90     |
