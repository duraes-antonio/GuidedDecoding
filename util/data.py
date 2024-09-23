from typing import Literal, List, Dict, Tuple, Union

from torch import Tensor

PyTorchDevice = Literal['cpu', 'cuda']


def unpack_and_move(device: PyTorchDevice, data: Union[List, Tuple, Dict]) -> Tuple[Tensor, Tensor]:
    if isinstance(data, (tuple, list)):
        image = data[0].to(device, non_blocking=True)
        gt = data[1].to(device, non_blocking=True)
        return image, gt

    if isinstance(data, dict):
        image = data['image'].to(device, non_blocking=True)
        gt = data['depth'].to(device, non_blocking=True)
        return image, gt

    raise Exception('Type not supported')
