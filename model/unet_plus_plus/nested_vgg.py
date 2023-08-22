from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.unet_plus_plus.blocks import LuizConvBlock, VGGBlock
from model.unet_plus_plus.encoders import LuizVGG16Encoder, ChatGPTVGG16BNEncoder


def concat(tensor: Tensor, concat_with: List[Tensor]):
    inter = F.interpolate(
        tensor,
        size=[concat_with[0].size(2), concat_with[0].size(3)],
        mode='bilinear',
        align_corners=True
    )
    concat_with.append(inter)
    return torch.cat(concat_with, dim=1)


class ChatGPTNestedUNetVGG16BNWFU(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()
        self.encoder = ChatGPTVGG16BNEncoder()
        nb_filter = [64, 128, 256, 512, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x: Tensor):
        feats = [x] + self.encoder(x)

        x0_0 = self.conv0_0(feats[0])

        x1_0 = self.conv1_0(feats[1])
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(feats[2])
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(feats[3])
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(feats[4])
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class ChatGPTNestedUNetVGG16BN(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()
        self.encoder = ChatGPTVGG16BNEncoder()
        nb_filter = [64, 128, 256, 512, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x: Tensor):
        feats = self.encoder(x)

        x0_0 = feats[0]

        x1_0 = feats[1]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = feats[2]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = feats[3]
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = feats[4]
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.up(x0_4)
        return self.final(output)


# Luiz encoder
class NestedUNetVGGLE(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        self.encoder = LuizVGG16Encoder()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x: Tensor):
        features = self.encoder(x)
        feature_indexes = [0, 5, 10, 17, 30]
        feats = [features[i] for i in feature_indexes]

        x0_0 = self.conv0_0(feats[0])

        x1_0 = feats[1]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = feats[2]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = feats[3]
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = feats[4]
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


# Uso de bloco com Leaky relu
class NestedUNetVGGLuizBlock(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        nb_filter = [input_channels, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.encoder = LuizVGG16Encoder()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = LuizConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = LuizConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = LuizConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = LuizConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = LuizConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = LuizConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = LuizConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        self.conv3_2 = LuizConvBlock(nb_filter[3] * 2 + nb_filter[4], nb_filter[3])

        self.conv0_3 = LuizConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = LuizConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        self.conv2_3 = LuizConvBlock(nb_filter[2] * 3 + nb_filter[3], nb_filter[2])

        self.conv0_4 = LuizConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.final = LuizConvBlock(nb_filter[0], num_classes)  # point-wise

    def forward(self, x: Tensor):
        features = self.encoder(x)
        feature_indexes = [0, 5, 10, 17, 23]
        feats = [features[i] for i in feature_indexes]

        x0_0 = feats[0]

        x1_0 = feats[1]
        x0_1 = self.conv0_1(concat(self.up(x1_0), [x0_0]))

        x2_0 = feats[2]
        x1_1 = self.conv1_1(concat(self.up(x2_0), [x1_0]))
        x0_2 = self.conv0_2(concat(self.up(x1_1), [x0_0, x0_1]))

        x3_0 = feats[3]
        x2_1 = self.conv2_1(concat(self.up(x3_0), [x2_0]))
        x1_2 = self.conv1_2(concat(self.up(x2_1), [x1_0, x1_1]))
        x0_3 = self.conv0_3(concat(self.up(x1_2), [x0_0, x0_1, x0_2]))

        x4_0 = feats[4]
        x3_1 = self.conv3_1(concat(self.up(x4_0), [x3_0]))
        x2_2 = self.conv2_2(concat(self.up(x3_1), [x2_0, x2_1]))
        x1_3 = self.conv1_3(concat(self.up(x2_2), [x1_0, x1_1, x1_2]))
        x0_4 = self.conv0_4(concat(self.up(x1_3), [x0_0, x0_1, x0_2, x0_3]))

        output = self.final(x0_4)
        return output
