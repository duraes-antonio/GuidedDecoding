from model.Unets.Blocks import *
from model.Unets.Encoder import Encoder


# codigo base: https://github.com/4uiiurz1/pytorch-nested-unet/blob/557ea02f0b5d45ec171aae2282d2cd21562a633e/archs.py


#                        bz, ch, he, wi
# feature[0]: torch.Size([12, 3, 240, 320]) -
# feature[1]: torch.Size([12, 32, 120, 160])
# feature[2]: torch.Size([12, 16, 120, 160]) -
# feature[3]: torch.Size([12, 24, 60, 80]) -
# feature[4]: torch.Size([12, 40, 30, 40]) -
# feature[5]: torch.Size([12, 80, 15, 20])
# feature[6]: torch.Size([12, 112, 15, 20]) -
# feature[7]: torch.Size([12, 192, 8, 10])
# feature[8]: torch.Size([12, 320, 8, 10])
# feature[9]: torch.Size([12, 1280, 8, 10])-


def concat(input, concat_with):
    inter = F.interpolate(input, size=[concat_with[0].size(2), concat_with[0].size(3)], mode='bilinear',
                          align_corners=True)
    concat_with.append(inter)
    concat = torch.cat(concat_with, dim=1)
    return concat


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=True, **kwargs):
        super().__init__()

        self.deep_supervision = deep_supervision

        nb_filter = [3, 16, 24, 40, 112, 1280]

        # self.upConcat = Up_concat()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder = Encoder()

        # U-net normal
        # self.conv5 = ConvBlock_BottleNeck(nb_filter[5]+nb_filter[4], nb_filter[4])
        # self.conv4 = ConvBlock_BottleNeck(nb_filter[4]+nb_filter[3], nb_filter[3])
        # self.conv3 = ConvBlock_BottleNeck(nb_filter[3]+nb_filter[2], nb_filter[2])
        # self.conv2 = ConvBlock_BottleNeck(nb_filter[2]+nb_filter[1], nb_filter[1])
        # self.conv1 = ConvBlock_BottleNeck(nb_filter[1]+nb_filter[0], nb_filter[0])

        # self.final = ConvBlock_BottleNeck(nb_filter[0], num_classes)

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv4_1 = ConvBlock(nb_filter[4] + nb_filter[5], nb_filter[4])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        self.conv3_2 = ConvBlock(nb_filter[3] * 2 + nb_filter[4], nb_filter[3])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        self.conv2_3 = ConvBlock(nb_filter[2] * 3 + nb_filter[3], nb_filter[2])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.conv1_4 = ConvBlock(nb_filter[1] * 4 + nb_filter[2], nb_filter[1])

        self.conv0_5 = ConvBlock(nb_filter[0] * 5 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = ConvBlock(nb_filter[0], num_classes)
            self.final2 = ConvBlock(nb_filter[0], num_classes)
            self.final3 = ConvBlock(nb_filter[0], num_classes)
            self.final4 = ConvBlock(nb_filter[0], num_classes)
            self.final5 = ConvBlock(nb_filter[0], num_classes)
        else:
            self.final = ConvBlock(nb_filter[0], num_classes)  # point-wise

    def forward(self, input):

        features = self.encoder(input)

        layers = [0, 2, 3, 4, 6, 9]
        feats = [features[layers[0]], features[layers[1]], features[layers[2]], features[layers[3]],
                 features[layers[4]], features[layers[5]]]

        # U-net normal
        # x = self.conv5(self.upConcat(feats[5],[feats[4]]))
        # x = self.conv4(self.upConcat(x,[feats[3]]))
        # x = self.conv3(self.upConcat(x,[feats[2]]))
        # x = self.conv2(self.upConcat(x,[feats[1]]))
        # x = self.conv1(self.upConcat(x,[feats[0]]))
        # output = self.final(x)

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

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output5 = self.final5(x0_5)
            return [output1, output2, output3, output4, output5]

        else:
            output = self.final(x0_5)
            return output
