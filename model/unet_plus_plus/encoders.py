import torch.nn as nn
from torchvision import models


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class LuizInceptionEncoder(nn.Module):
    def __init__(self):
        super(LuizInceptionEncoder, self).__init__()
        self.inception_model = models.inception_v3(pretrained=True)
        self.encoder = nn.Sequential(*list(self.inception_model.children())[:-2])

    def forward(self, x):
        enc_features = [x]

        for layer in self.encoder:
            x = layer(x)
            enc_features.append(x)
        return enc_features
