from torch import nn as nn
from torchvision import models


class LuizResNetEncoder(nn.Module):
    def __init__(self):
        super(LuizResNetEncoder, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        encoder_layers = [
            resnet101.conv1,
            resnet101.bn1,
            resnet101.relu,
            resnet101.maxpool,
            resnet101.layer1,
            resnet101.layer2,
            resnet101.layer3,
            resnet101.layer4
        ]
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        enc_features = [x]
        for layer in self.encoder:
            x = layer(x)
            enc_features.append(x)
        return enc_features
