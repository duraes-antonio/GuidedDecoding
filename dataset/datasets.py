from typing import Literal

import torch
from torch.utils.data import DataLoader

from config import SEED
from dataset.nyu_reduced import get_NYU_dataset
from options.dataset_resolution import Resolutions
from reproducibility import set_seed_worker

"""
Preparation of dataloaders for Datasets
"""

DatasetSplitOptions = Literal["train", "test"]


def get_dataset(
        path: str,
        split: DatasetSplitOptions,
        resolution: Resolutions = Resolutions.Full,
        uncompressed=False,
):
    return get_NYU_dataset(
        path, split, resolution=resolution, uncompressed=uncompressed
    )


def get_dataloader(
        dataset_name,
        path: str,
        split: DatasetSplitOptions,
        resolution: Resolutions = Resolutions.Full,
        batch_size=4,
        workers=0,
        uncompressed=False,
        shuffle=True,
):
    if dataset_name == "nyu_reduced":
        dataset = get_NYU_dataset(
            path, split, resolution=resolution, uncompressed=uncompressed
        )

    else:
        raise Exception(f"Dataset not existent: '{dataset_name}'")

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=set_seed_worker,
        generator=dataloader_generator,
    )

