import torch
from torch.utils.data import DataLoader

from config import SEED
from dataset.nyu_reduced import get_NYU_dataset
from options.dataset_resolution import Resolutions
from reproducibility import set_seed_worker

"""
Preparation of dataloaders for Datasets
"""


def get_dataloader(dataset_name,
                   path,
                   split='train',
                   resolution: Resolutions = Resolutions.Full,
                   augmentation='alhashim',
                   interpolation='linear',
                   batch_size=4,
                   workers=0,
                   uncompressed=False):
    if dataset_name == 'nyu_reduced':

        # dataset = DepthEstimationDataset(read_nyu_csv(path), transform=transform_swin)
        dataset = get_NYU_dataset(path, split, resolution=resolution, uncompressed=uncompressed)
    else:
        print('Dataset not existent')
        exit(0)

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=set_seed_worker,
        generator=dataloader_generator,
    )
    return dataloader
