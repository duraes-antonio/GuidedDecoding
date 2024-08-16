from typing import Dict, Any, OrderedDict, Optional, Tuple

import torch
from torch import nn
from torch.nn import Identity, ELU
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler, StepLR

from model import loader
from options.dataset_resolution import Resolutions
from options.model import Models
from options.task import Task
from util.config import DEVICE


def __get_output_channels__(from_task: Task) -> int:
    return 1 if from_task == Task.DEPTH else 19


def __get_last_layer__(task: Task) -> nn.Module:
    out_channels = __get_output_channels__(task)
    return nn.Sequential(
        nn.Conv2d(16, out_channels, (3, 3), (1, 1), (1, 1)),
    ).to(DEVICE)


class ModelLoader:
    def __init__(
            self,
            target_model: Models,
            resolution: Resolutions,
            checkpoint_load_path: str,
            from_task: Task,
            to_task: Task
    ):
        out_channels = __get_output_channels__(from_task)
        scheduler_step_size = 15
        learning_rate = 1e-4
        self.optimizer = None
        self.scheduler = None
        self.model: nn.Module = loader.load_model(target_model, resolution, num_classes=out_channels)

        if checkpoint_load_path:
            checkpoint = torch.load(checkpoint_load_path, map_location=DEVICE)

            if self.__is_from_lightning__(checkpoint):
                self.model = self.__load_from_lightning__(checkpoint, self.model, self.optimizer, self.scheduler)
                print(f'--> Loaded from lightning checkpoint:\nPATH: {checkpoint_load_path}')

            else:
                self.model.load_state_dict(torch.load(checkpoint_load_path))
                print(f'--> Loaded from MODEL checkpoint:\nPATH: {checkpoint_load_path}')

                # m, o, s = self.__load_from_guided__(checkpoint, self.model, self.optimizer, self.scheduler)
                # self.model = m
                # self.optimizer = o
                # self.scheduler = s
                # print(f'--> Loaded from GUIDED checkpoint:\nPATH: {checkpoint_load_path}')

            new_last_layer = __get_last_layer__(to_task)
            self.model.segmentation_head[0] = new_last_layer

        self.optimizer = Adam(self.model.parameters(), learning_rate)
        self.scheduler = StepLR(self.optimizer, scheduler_step_size, 0.1)

    @staticmethod
    def __is_from_lightning__(checkpoint: Dict) -> bool:
        if 'state_dict' not in checkpoint:
            return False

        state: Dict[str, Any] = checkpoint['state_dict']
        return any(k.startswith('model.') for k in state.keys())

    @staticmethod
    def __load_from_lightning__(
            checkpoint: Dict,
            model: nn.Module,
            optimizer: Optional[Optimizer] = None,
            lr_scheduler: Optional[LRScheduler] = None
    ) -> nn.Module:
        model_state = checkpoint['state_dict']
        model_state = {k.replace('model.', ''): v for k, v in model_state.items()}
        encoder_weights = {
            k.replace('encoder.', ''): v
            for k, v in model_state.items()
            if k.startswith('encoder.')
        }
        decoder_weights = {
            k.replace('decoder.', ''): v
            for k, v in model_state.items()
            if k.startswith('decoder.')
        }
        model.encoder.load_state_dict(encoder_weights)
        model.decoder.load_state_dict(decoder_weights)

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_states"][-1])
        return model

    @staticmethod
    def __load_from_guided__(
            checkpoint: Dict,
            model: nn.Module,
            optimizer: Optional[Optimizer] = None,
            lr_scheduler: Optional[LRScheduler] = None
    ) -> Tuple[nn.Module, Optimizer, Optional[LRScheduler]]:
        stored_model: OrderedDict = checkpoint["model"]

        def fix_key(key: str, model_dict: OrderedDict) -> OrderedDict:
            if key in model_dict:
                new_key = key.replace('.0.0.', '.0.')
                model_dict[new_key] = model_dict[key]
                del model_dict[key]
            return model_dict

        fix_key('segmentation_head.0.0.weight', stored_model)
        fix_key('segmentation_head.0.0.bias', stored_model)

        model.load_state_dict(stored_model)

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        return model, optimizer, lr_scheduler
