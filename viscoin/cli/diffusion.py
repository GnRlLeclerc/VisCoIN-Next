import click
import torch

from viscoin.cli.utils import device, dataset, viscoin_pickle_path
from viscoin.models.utils import load_viscoin_pickle

from viscoin.datasets.cub import CUB_200_2011
from viscoin.datasets.funnybirds import FunnyBirds

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter

from matplotlib import pyplot as plt

from PIL import Image
import os


@click.command()
@dataset
@device
@viscoin_pickle_path
@click.option(
    "--concept2clip-path",
    default="checkpoints/cub/concept2clip1024.pt",
    help="The path to the Concept2CLIP model weights",
)
def image_to_prompt_diffusion(
    dataset: str,
    device: str,
    viscoin_pickle_path: str,
    concept2clip_path: str,
):
    """Takes an image and gets its clip embedding using Concept2CLIP.
    Then this embedding is given as a prompt to the diffusion model.

    Args:
        image_path (str): The path of the image to be converted to a prompt.
    """

    match dataset:
        case "cub":
            dataset = CUB_200_2011(mode="test")
        case "funnybirds":
            dataset = FunnyBirds()
        case _:
            raise ValueError(f"Unsupported dataset: {dataset}")

    torch.manual_seed(42)

    # Load the image
    random_index = torch.randint(0, len(dataset), (1,)).item()
    original_image = Image.open(
        os.path.join(dataset.dataset_path, "images", dataset.image_paths[random_index])
    )

    image = dataset.transform(original_image).unsqueeze(0).to(device)

    # Load the models
    viscoin = load_viscoin_pickle(viscoin_pickle_path)
    classifier = viscoin.classifier.to(device)
    concept_extractor = viscoin.concept_extractor.to(device)

    concept2clip = torch.load(concept2clip_path, weights_only=False).to(device)

    # Get the CLIP embedding of the image from its VisCoIN concepts
    classes, hidden_states = classifier.forward(image)
    encoded_concepts, extra_info = concept_extractor.forward(hidden_states[-3:])
    clip_embeddings = concept2clip.forward(encoded_concepts)

    base_model_path = "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = "checkpoints/ip-adapter_sd15.bin"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    # generate image variations
    images = ip_model.generate(
        pil_image=original_image, num_samples=1, num_inference_steps=50, seed=42
    )

    images_concept2clip = ip_model.generate(
        clip_image_embeds=clip_embeddings,
        num_samples=1,
        num_inference_steps=50,
        seed=42,
    )

    print(images)

    # plot original and output image
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(original_image)
    axs[0].axis("off")
    axs[0].set_title("Original Image")
    axs[1].imshow(images[0])
    axs[1].axis("off")
    axs[1].set_title("Output Image")
    axs[2].imshow(images_concept2clip[0])
    axs[2].axis("off")
    axs[2].set_title("Output Image (C2C)")
    plt.tight_layout()
    # plt.show()

    dirs = [56]
    lambdas = [0, 5, 10, 20, 30, 40]
    images = {i: {} for i in lambdas}
    for dir in dirs:
        for i, lambda_ in enumerate(lambdas):
            # Amplify the concept dir by lambda_
            amplified_concepts = encoded_concepts.clone()
            amplified_concepts[0][dir] *= lambda_

            # Get the CLIP embedding of the image from its VisCoIN concepts
            clip_embeddings = concept2clip.forward(amplified_concepts)

            # Generate the image
            images[lambda_] = ip_model.generate(
                clip_image_embeds=clip_embeddings,
                num_samples=1,
                num_inference_steps=50,
                seed=42,
            )

        # Plot an image grid with all these images
        fig, axs = plt.subplots(1, len(lambdas), figsize=(20, 20))
        for i, lambda_ in enumerate(lambdas):
            axs[i].imshow(images[lambda_][0])
            axs[i].axis("off")
            axs[i].set_title(f"Lambda {lambda_}", fontsize=20)
        plt.tight_layout()
    plt.show()
