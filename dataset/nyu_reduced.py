import csv
import os
from io import BytesIO
from pathlib import Path
from random import shuffle
from typing import List, Tuple
from zipfile import ZipFile

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from dataset.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor
from options.dataset_resolution import shape_by_resolution, Resolutions


class DepthDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], split, transform=None):
        self.data = data
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(Path(sample[0]))
        depth = Image.open(Path(sample[1]))
        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        if self.split == 'train':
            depth = depth / 255.0 * 10.0  # From 8bit to range [0, 10] (meter)
        elif self.split == 'val':
            depth = depth * 0.001

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        return (sample['image'], sample['depth'])

    def __len__(self):
        return len(self.data)


class NYU_Testset_Extracted(Dataset):
    def __init__(self, root, resolution=Resolutions.Full):
        self.root = root
        self.resolution = shape_by_resolution[resolution]
        self.files = os.listdir(self.root)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.files[index])

        data = np.load(image_path)
        depth, image = data['depth'], data['image']
        depth = np.expand_dims(depth, axis=2)

        image, depth = data['image'], data['depth']
        image = np.array(image)
        depth = np.array(depth)
        return image, depth

    def __len__(self):
        return len(self.files)


class NYU_Testset(Dataset):
    def __init__(self, zip_path):
        input_zip = ZipFile(zip_path)
        data = {name: input_zip.read(name) for name in input_zip.namelist()}
        self.data = data

    def __getitem__(self, idx):
        key_item = list(self.data.keys())[idx]
        data_numpy = np.load(BytesIO(self.data[key_item]))
        image = torch.from_numpy(data_numpy['image']).type(torch.float32)
        depth = torch.from_numpy(data_numpy['depth']).type(torch.float32)
        return image, depth

    def __len__(self):
        return len(self.data)


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (row.split(',') for row in (data['dataset/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list(
        (row.split(',') for row in (data['dataset/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    # Debugging
    # if True: nyu2_train = nyu2_train[:100]
    # if True: nyu2_test = nyu2_test[:100]

    print('Loaded (Train Images: {0}, Test Images: {1}).'.format(len(nyu2_train), len(nyu2_test)))
    return data, nyu2_train, nyu2_test


def train_transform(resolution: Tuple[int, int]):
    transform = transforms.Compose([
        Resize(resolution),
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor(test=False, maxDepth=10.0)
    ])
    return transform


def val_transform(resolution):
    transform = transforms.Compose([
        Resize(resolution),
        ToTensor(test=True, maxDepth=10.0)
    ])
    return transform


def get_NYU_dataset(zip_path, split, resolution: Resolutions, uncompressed=False):
    final_size = shape_by_resolution[resolution]

    if split != 'test':
        data = read_nyu_csv(zip_path)
        shuffle(data)

        if split == 'train':
            transform = train_transform(final_size)
            return DepthDataset(data, split, transform=transform)

    elif split == 'test':
        if uncompressed:
            dataset = NYU_Testset_Extracted(zip_path)
        else:
            dataset = NYU_Testset(zip_path)

    return dataset


def read_nyu_csv(csv_file_path) -> List[Tuple[str, str]]:
    """
    Lê CSV que relacionada x e y e retona uma lista de pares de paths (x, y)
    :param csv_file_path: Path do arquivo CSV com o nome de x e y
    :return: Lista de pares (path input, path ground truth)
    """
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        return [('./' + row[0], './' + row[1]) for row in csv_reader if len(row) > 0]
