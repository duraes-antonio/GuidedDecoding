from typing import List

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn, Tensor


class CoatNet2SMP(nn.Module, EncoderMixin):

    @property
    def out_channels(self):
        return self._out_channels

    def __init__(self, **params):
        super().__init__()
        model_name_by_size = {
            (3, 224, 224): 'coatnet_rmlp_2_rw_224.sw_in1k',
            (3, 384, 384): 'timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
        }
        self.model = timm.create_model(
            model_name_by_size[params['input_size']],
            pretrained=True,
            features_only=True,
        )
        self._out_channels: List[int] = [3, 128, 128, 256, 512, 1024]
        self._depth: int = 5
        self._in_channels: int = 3

    def forward(self, x: Tensor) -> List[Tensor]:
        b, c, h, w = x.shape
        dummy = torch.empty([b, 0, h, w], dtype=x.dtype, device=x.device)
        out = self.model(x)
        out = [dummy] + out
        return out


class CoatNet3SMP(nn.Module, EncoderMixin):

    @property
    def out_channels(self):
        return self._out_channels

    def __init__(self, **params):
        super().__init__()
        self.model = timm.create_model(
            'coatnet_3_rw_224.sw_in12k',
            pretrained=True,
            features_only=True,
        )

        self._out_channels: List[int] = [0, 192, 192, 384, 768, 1536]
        self._depth: int = 5
        self._in_channels: int = 3

    def forward(self, x: Tensor) -> List[Tensor]:
        b, c, h, w = x.shape
        dummy = torch.empty([b, 0, h, w], dtype=x.dtype, device=x.device)
        out = self.model(x)
        out = [dummy] + out
        return out


class CoatLiteMediumSMP(nn.Module, EncoderMixin):

    @property
    def out_channels(self):
        return self._out_channels

    def __init__(self, **params):
        super().__init__()
        model_name_by_size = {
            (3, 224, 224): 'coat_lite_medium.in1k',
            (3, 384, 384): 'coat_lite_medium_384.in1k',
        }
        self.model = timm.create_model(
            model_name_by_size[params['input_size']],
            pretrained=True,
            num_classes=0,
            return_interm_layers=True,
            out_features=['x1_nocls', 'x2_nocls', 'x3_nocls', 'x4_nocls'],
        )
        self._out_channels: List[int] = [3, 3, 128, 256, 320, 512]
        self._depth: int = 5
        self._in_channels: int = 3

    def forward(self, x: Tensor) -> List[Tensor]:
        b, c, h, w = x.shape
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        # dummy = torch.empty([b, 0, h // 2, w // 2], dtype=x.dtype, device=x.device)
        out = self.model.forward_features(x)
        out = [out[key] for key in out.keys()]
        out = [x, x_half] + out
        return out


def __register_smp_custom_encoders__():
    smp.encoders.encoders["coatnet-2_224"] = {
        "encoder": CoatNet2SMP,
        "pretrained_settings": {},
        "params": {
            'input_size': (3, 224, 224),
        },
    }
    smp.encoders.encoders["coatnet-2_384"] = {
        "encoder": CoatNet2SMP,
        "pretrained_settings": {},
        "params": {
            'input_size': (3, 384, 384),
        },
    }

    smp.encoders.encoders["coatnet-3_224"] = {
        "encoder": CoatNet3SMP,
        "pretrained_settings": {},
        "params": {},
    }

    smp.encoders.encoders["coat-lite-medium_224"] = {
        "encoder": CoatLiteMediumSMP,
        "pretrained_settings": {},
        "params": {
            'input_size': (3, 224, 224),
        },
    }
    smp.encoders.encoders["coat-lite-medium_384"] = {
        "encoder": CoatLiteMediumSMP,
        "pretrained_settings": {},
        "params": {
            'input_size': (3, 384, 384),
        },
    }


__register_smp_custom_encoders__()
