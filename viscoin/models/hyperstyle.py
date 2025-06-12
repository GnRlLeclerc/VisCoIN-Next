"""
Helper functions for the HyperStyle model.
"""

import os
import sys
from argparse import Namespace

import gdown
import torch
import torchvision.transforms.v2 as transforms
from torch import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "hyperstyle"))

from hyperstyle.models.hyperstyle import HyperStyle
from hyperstyle.utils.inference_utils import run_inversion
from hyperstyle.utils.model_utils import load_model

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def _cache_path():
    """Download the HyperStyle model to the same cache as HugginFace,
    in case the user has moved the cache location for disk quota reasons."""

    HF_CACHE = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(HF_CACHE, "hyperstyle")


def load_hyperstyle(path=_cache_path()) -> tuple[HyperStyle, Namespace]:
    """Load the HyperStyle model for FFHQ."""

    if not os.path.exists(path):
        os.makedirs(path)
        download_hyperstyle(path)

    model_path = os.path.join(path, "hyperstyle_ffhq.pt")
    net, opts = load_model(
        model_path,
        update_opts={"w_encoder_checkpoint_path": os.path.join(path, "faces_w_encoder.pt")},
    )

    opts.n_iters_per_batch = 5
    opts.resize_outputs = False  # generate outputs at full resolution

    return net, opts  # type: ignore


def compute_latents(
    model: HyperStyle, opts: Namespace, image: Tensor
) -> tuple[Tensor, list[Tensor | None]]:
    """Compute the latent code and weight shifts for an image.

    Args:
        model: HyperStyle model.
        opts: Namespace with options for the model.
        image: Input image tensor of shape (3, 256, 256).

    Returns:
        code: (batch_size, 18, 512) tensor representing the W+ code.
        result_deltas: List of weight deltas for the decoder.
    """

    with torch.no_grad():
        # Run the inversion to get the latent code and weight shifts
        _, _, result_deltas, code = run_inversion(  # type: ignore
            image.unsqueeze(0).cuda(),
            model,
            opts,
        )

    return code, result_deltas  # type: ignore


def generate(model: HyperStyle, w_plus: Tensor, weights_deltas: list[Tensor | None]) -> Tensor:
    """Generate an image from W+ code and weight deltas.

    Args:
        model: HyperStyle model.
        w_plus (18, 512): W+ code tensor.
        weights_deltas: TODO shape, the weight deltas for the decoder
    """
    decoder = model.decoder

    image, _ = decoder.forward(
        w_plus,
        weights_deltas=weights_deltas,
        randomize_noise=False,
        input_is_latent=True,
    )

    return image


def download_hyperstyle(path: str):
    """Download the FFHQ variant of HyperStyle model.
    All checkpoints are downloaded to the specified path."""

    hyperstyle_id = "1C3dEIIH1y8w1-zQMCyx7rDF0ndswSXh4"
    w_encoder_id = "1M-hsL3W_cJKs77xM1mwq2e9-J0_m7rHP"
    ffhq_stylegan_id = "1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT"

    def make_url(file_id: str) -> str:
        return f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        make_url(hyperstyle_id),
        output=os.path.join(path, "hyperstyle_ffhq.pt"),
    )
    gdown.download(make_url(w_encoder_id), output=os.path.join(path, "faces_w_encoder.pt"))
    gdown.download(make_url(ffhq_stylegan_id), output=os.path.join(path, "ffhq_stylegan.pt"))
