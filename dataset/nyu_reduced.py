import csv
import os
from io import BytesIO
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Union
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
        sample = (
            sample[0].replace("./", "/home/antonio/Área de trabalho/seg_depth/"),
            sample[1].replace("./", "/home/antonio/Área de trabalho/seg_depth/"),
        )
        image = Image.open(Path(sample[0]))
        depth = Image.open(Path(sample[1]))
        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        if self.split == "train":
            depth = depth / 255.0 * 10.0  # From 8bit to range [0, 10] (meter)
        elif self.split == "val":
            depth = depth * 0.001

        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["depth"]

    def __len__(self):
        return len(self.data)


class NYU_Testset(Dataset):
    def __init__(self, zip_path, percent=1):
        input_zip = ZipFile(zip_path)
        name_list = input_zip.namelist()
        name_list = name_list[: int(len(name_list) * percent)]
        data = {name: input_zip.read(name) for name in name_list}
        self.data = data
        del input_zip

    def __getitem__(self, idx):
        key_item = list(self.data.keys())[idx]
        data_numpy = np.load(BytesIO(self.data[key_item]))
        image = torch.from_numpy(data_numpy["image"]).type(torch.float32)
        depth = torch.from_numpy(data_numpy["depth"]).type(torch.float32)
        return image, depth

    def __len__(self):
        return len(self.data)


def train_transform(resolution: Tuple[int, int]):
    transform = transforms.Compose(
        [
            Resize(resolution),
            RandomHorizontalFlip(),
            RandomChannelSwap(0.5),
            ToTensor(test=False, max_depth=10.0),
        ]
    )
    return transform


def test_transform(resolution: Tuple[int, int], max_depth=10.0):
    import torchvision
    crop = [20, 460, 24, 616]
    downscale_image = torchvision.transforms.Resize(resolution)
    to_tensor = ToTensor(test=True, max_depth=max_depth)
    return transforms.Compose(
        [
        ]
    )


def val_transform(resolution):
    transform = transforms.Compose(
        [Resize(resolution), ToTensor(test=True, max_depth=10.0)]
    )
    return transform


def get_NYU_dataset(
        data_path: Union[str, Path], split, resolution: Resolutions, uncompressed=False
):
    final_size = shape_by_resolution[resolution]

    if split != "test":
        data = read_nyu_csv(data_path)

        if split == "train":
            transform = train_transform(final_size)
            return DepthDataset(data, split, transform=transform)

    elif split == "test":
        dataset = NYU_Testset(data_path)

    return dataset


def read_nyu_csv(csv_file_path) -> List[Tuple[str, str]]:
    """
    Lê CSV que relacionada x e y e retona uma lista de pares de paths (x, y)
    :param csv_file_path: Path do arquivo CSV com o nome de x e y
    :return: Lista de pares (path input, path ground truth)
    """
    with open(csv_file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        return [("./" + row[0], "./" + row[1]) for row in csv_reader if len(row) > 0]
