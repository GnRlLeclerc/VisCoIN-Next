"""
Utilities to download, unzip datasets.
Because some of our datasets are on Kaggle, for ease of use when changing the destination
of datasets in limited disk space machines, we download datasets that are not on Kaggle
into the Kaggle cache path with username "viscoin".
"""

import os
import zipfile

import kagglehub
import requests
from tqdm import tqdm


def download(url: str):
    """Download a ZIP file from a URL and extract it to the specified destination."""

    output_filename = "temp.zip"

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192
    t = tqdm(total=total_size, unit="i", unit_scale=True, desc=output_filename)

    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            t.update(len(chunk))
    t.close()

    # Unzip the downloaded file
    print("Unzipping dataset...")
    with zipfile.ZipFile(output_filename, "r") as zip_ref:
        zip_ref.extractall(dataset_path(""))

    # Remove the zip file after extraction
    os.remove(output_filename)

    print("Done.")


def dataset_exists(name: str) -> bool:
    """Check if a dataset has already been downloaded."""

    path = dataset_path(name)

    return os.path.exists(path) and os.path.isdir(path)


def dataset_path(name: str) -> str:
    """Get the full path to a custom downloaded dataset, inside the Kagglehub cache path."""

    kaggle_cache = kagglehub.config.DEFAULT_CACHE_FOLDER  # type: ignore

    return os.path.join(kaggle_cache, "datasets", "viscoin", name)
