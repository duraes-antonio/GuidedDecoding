from torch import nn as nn
from torchvision import models


class LuizVGG16Encoder(nn.Module):
    def __init__(self):
        super(LuizVGG16Encoder, self).__init__()
        self.encoder = models.vgg16(pretrained=True)
        freeze = False

        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        features = [x]
        for _, v in self.encoder.features._modules.items():
            features.append(v(features[-1]))
        return features


class LuizVGG16BNEncoder(nn.Module):
    def __init__(self):
        super(LuizVGG16BNEncoder, self).__init__()
        self.encoder = models.vgg16_bn(pretrained=True)

    def forward(self, x):
        features = [x]
        for _, v in self.encoder.features._modules.items():
            features.append(v(features[-1]))
        return features


class ChatGPTVGG16BNEncoder(nn.Module):
    def __init__(self):
        super(ChatGPTVGG16BNEncoder, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        encoder_blocks = [
            vgg16_bn.features[:7],  # conv1
            vgg16_bn.features[7:14],  # conv2
            vgg16_bn.features[14:24],  # conv3
            vgg16_bn.features[24:34],  # conv4
            vgg16_bn.features[34:],  # conv5
        ]
        self.encoder = nn.ModuleList(encoder_blocks)

    def forward(self, x):
        encoder_outputs = []
        for block in self.encoder:
            x = block(x)
            encoder_outputs.append(x)

        return encoder_outputs
