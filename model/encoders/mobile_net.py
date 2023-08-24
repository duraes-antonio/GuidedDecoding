from torch import nn as nn
from torchvision import models

from model.util import count_parameters


class LuizMobileNetV2Encoder(nn.Module):
    def __init__(self):
        super(LuizMobileNetV2Encoder, self).__init__()
        backbone_nn = models.mobilenet_v2(pretrained=True)
        freeze = False

        for param in backbone_nn.parameters():
            param.requires_grad = not freeze  # precisa ser False para congelar

        count_parameters(backbone_nn)
        self.original_model = backbone_nn

    def forward(self, x):
        features = [x]
        for _, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features
