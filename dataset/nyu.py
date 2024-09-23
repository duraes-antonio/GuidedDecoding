from __future__ import annotations

import csv
import os
from io import BytesIO
from os import path
from pathlib import Path
from typing import List, Tuple, Dict
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transforms import RandomHorizontalFlip, RandomChannelSwap, ToTensor, Resize, PairToDict, ResizeOnlyImage


# TODO: Substituir por generator?
def read_nyu_csv(
        csv_file_path: str,
        repository_path: str = "/home/antonio/Área de trabalho/seg_depth/"
) -> List[Tuple[str, str]]:
    """
    Lê CSV que relacionada x e y e retorna uma lista de pares de paths (x, y)
    :param csv_file_path: Path do arquivo CSV com o nome de x e y
    :param repository_path: Path do repositório que está executando o código
    :return: Lista de pares (path input, path ground truth)
    """
    with open(csv_file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        return [
            (path.join(repository_path, row[0]), path.join(repository_path, row[1]))
            for row in csv_reader
            if len(row) > 0
        ]


class NyuDataset(Dataset):
    def __init__(
            self,
            pairs_path: List[Tuple[str, str]],
            split="train",
            transform: nn.Module = None,
    ):
        self.data_paths = pairs_path
        self.n_data = len(self.data_paths)
        self.transform = transform
        self.split = split

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor] | Tuple[np.ndarray, np.ndarray]:
        input_path, gt_path = self.data_paths[index]
        image = Image.open(Path(input_path))
        depth = Image.open(Path(gt_path))
        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        # From 8bit to range [0, 10] (meter)
        if self.split == "train":
            depth = depth / 255.0 * 10.0

        elif self.split == "val":
            depth = depth * 0.001

        # TODO: Ajustar para não usar dicionário
        if self.transform:
            sample = self.transform({"image": image, "depth": depth})
            image = sample["image"]
            depth = sample["depth"]

        return image, depth

    def __len__(self):
        return self.n_data


# TODO: Pensar em como unir tudo em uma classe de dataset apenas
class NyuNumpyZipDataset(Dataset):
    def __init__(self, zip_path: str, transform: nn.Module = None):

        if zip_path.endswith(".zip"):
            input_zip = ZipFile(zip_path)
            name_list = input_zip.namelist()
            data = {name: input_zip.read(name) for name in name_list}

        else:
            data = self.__folder_to_dict__(zip_path)

        self.data = data
        self.data_filenames = list(self.data.keys())
        self.transform = transform
        self.n_data = len(self.data)
        del input_zip

    @staticmethod
    def __folder_to_dict__(directory) -> Dict[str, str]:
        files_dict = {}

        for root, dirs, files in os.walk(directory):
            for filename in files:
                full_path = (os.path.join(root, filename))
                files_dict[filename] = full_path
        return files_dict

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        key_item = self.data_filenames[index]
        data_numpy = np.load(BytesIO(self.data[key_item]))
        image = torch.from_numpy(data_numpy["image"]).type(torch.float32)
        depth = torch.from_numpy(data_numpy["depth"]).type(torch.float32)

        if self.transform:
            sample = self.transform({"image": image, "depth": depth})
            image = sample["image"]
            depth = sample["depth"]

        return image, depth

    def __len__(self):
        return self.n_data


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
    return transforms.Compose(
        [
            PairToDict(),
            ToTensor(test=True, max_depth=max_depth),
            # UnsqueezeDict(),
            ResizeOnlyImage(resolution),
        ]
    )


def val_transform(resolution):
    transform = transforms.Compose(
        [Resize(resolution), ToTensor(test=True, max_depth=10.0)]
    )
    return transform
