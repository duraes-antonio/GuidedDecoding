from typing import Any, Tuple

import numpy
from PIL import Image
from numpy import ndarray
from torchvision.datasets import Cityscapes

from dataset.transforms_seg import transform

ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [
    ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33
]
class_names = [
    'unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)
colors = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]
label_colours = dict(zip(range(n_classes), colors))


def encode_segmap(mask) -> ndarray:
    # remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def decode_segmap(temp) -> ndarray:
    # convert gray scale to color
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = numpy.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


class CityscapesWrapper(Cityscapes):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        targets: Any = []

        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed = transform(image=numpy.array(image), mask=numpy.array(target))
        return transformed['image'], transformed['mask']


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import os


class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', mode='fine', transform=None, eval=False):
        self.transform = transform
        if mode == 'fine':
            self.mode = 'gtFine'
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        # Preparing a list of all labelTrainIds rgb and
        # ground truth images. Setting relabbelled=True is recommended.

        self.label_path = os.path.join(os.getcwd(), root_dir + '/' + self.mode + '/' + self.split)
        self.rgb_path = os.path.join(os.getcwd(), root_dir + '/leftImg8bit/' + self.split)
        city_list = os.listdir(self.label_path)
        for city in city_list:
            temp = os.listdir(self.label_path + '/' + city)
            list_items = temp.copy()

            # 19-class label items being filtered
            for item in temp:
                if not item.endswith('labelIds.png', 0, len(item)):
                    list_items.remove(item)

            # defining paths
            list_items = ['/' + city + '/' + path for path in list_items]

            self.yLabel_list.extend(list_items)
            self.XImg_list.extend(
                ['/' + city + '/' + path for path in os.listdir(self.rgb_path + '/' + city)]
            )

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        image = Image.open(self.rgb_path + self.XImg_list[index])
        y = Image.open(self.label_path + self.yLabel_list[index])

        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        image = transforms.ToTensor()(image)
        y = np.array(y)
        y = torch.from_numpy(y)

        y = y.type(torch.LongTensor)
        if self.eval:
            return image, y, self.XImg_list[index]
        else:
            return image, y