import torch.nn as nn
from torchvision import models


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LuizMobileNetV2Encoder(nn.Module):
    def __init__(self):
        super(LuizMobileNetV2Encoder, self).__init__()
        backbone_nn = models.mobilenet_v2(pretrained=True)
        freeze = True

        print("NOT freezing backbone layers - ", type(backbone_nn).__name__)
        for param in backbone_nn.parameters():
            param.requires_grad = not freeze  # precisa ser False para congelar

        count_parameters(backbone_nn)
        self.original_model = backbone_nn

    def forward(self, x):
        features = [x]
        for _, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class LuizVGG16Encoder(nn.Module):
    def __init__(self):
        super(LuizVGG16Encoder, self).__init__()
        self.encoder = models.vgg16()
        freeze = True

        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        features = [x]
        for _, v in self.encoder.features._modules.items():
            features.append(v(features[-1]))
        return features


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
