import random

import numpy
import torch
from torch.utils.data import DataLoader

from config import SEED
from dataset.nyu_reduced import get_NYU_dataset

torch.manual_seed(SEED)

"""
Preparation of dataloaders for Datasets
"""


def get_dataloader(dataset_name,
                   path,
                   split='train',
                   resolution='full',
                   augmentation='alhashim',
                   interpolation='linear',
                   batch_size=4,
                   workers=0,
                   uncompressed=False):
    if dataset_name == 'nyu_reduced':

        # dataset = DepthEstimationDataset(read_nyu_csv(path), transform=transform_swin)
        dataset = get_NYU_dataset(path, split, resolution=resolution, uncompressed=uncompressed)
    else:
        print('Dataset not existant')
        exit(0)

    def seed_worker(worker_id):
        numpy.random.seed(SEED)
        random.seed(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataloader
