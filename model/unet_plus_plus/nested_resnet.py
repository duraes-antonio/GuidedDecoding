from torch import nn, Tensor

from model.unet_plus_plus.blocks import LuizConvBlock
from model.unet_plus_plus.encoders import LuizResNetEncoder
from model.unet_plus_plus.nested import concat


class NestedUNetResNetLE(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        self.encoder = LuizResNetEncoder()
        nb_filter = [input_channels, 64, 256, 512, 1024, 2048]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = LuizConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = LuizConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = LuizConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = LuizConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv4_1 = LuizConvBlock(nb_filter[4] + nb_filter[5], nb_filter[4])

        self.conv0_2 = LuizConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = LuizConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = LuizConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        self.conv3_2 = LuizConvBlock(nb_filter[3] * 2 + nb_filter[4], nb_filter[3])

        self.conv0_3 = LuizConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = LuizConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        self.conv2_3 = LuizConvBlock(nb_filter[2] * 3 + nb_filter[3], nb_filter[2])

        self.conv0_4 = LuizConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.conv1_4 = LuizConvBlock(nb_filter[1] * 4 + nb_filter[2], nb_filter[1])

        self.conv0_5 = LuizConvBlock(nb_filter[0] * 5 + nb_filter[1], nb_filter[0])

        self.final = LuizConvBlock(nb_filter[0], num_classes)

    def forward(self, x: Tensor):
        features = self.encoder(x)
        feature_indexes = [0, 2, 5, 6, 7, 8]
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

        x5_0 = feats[5]
        x4_1 = self.conv4_1(concat(self.up(x5_0), [x4_0]))
        x3_2 = self.conv3_2(concat(self.up(x4_1), [x3_0, x3_1]))
        x2_3 = self.conv2_3(concat(self.up(x3_2), [x2_0, x2_1, x2_2]))
        x1_4 = self.conv1_4(concat(self.up(x2_3), [x1_0, x1_1, x1_2, x1_3]))
        x0_5 = self.conv0_5(concat(self.up(x1_4), [x0_0, x0_1, x0_2, x0_3, x0_4]))

        output = self.final(x0_5)
        return output
