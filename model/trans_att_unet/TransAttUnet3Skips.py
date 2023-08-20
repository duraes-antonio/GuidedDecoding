from typing import Dict, Tuple, List

from torch import Tensor

from .unet_parts import *
from .unet_parts_att_transformer import *


class UnetAttentionTransformerMultiscale3Skips(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UnetAttentionTransformerMultiscale3Skips, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(1024, 256 // factor, bilinear)
        self.up3 = Up(512 + 256 + 128 + 128, 128 // factor, bilinear)
        self.up4 = Up(256 + 256, 64, bilinear)
        self.outc = OutConv(128, n_classes)

        self.d1_d3 = nn.Sequential(
            DoubleConv(512, 128),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.d1_d4 = nn.Sequential(
            DoubleConv(512, 64),
            nn.UpsamplingBilinear2d(scale_factor=16)
        )
        self.d2_d4 = nn.Sequential(
            DoubleConv(256, 64),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.conv_by_origin_target_channels: Dict[Tuple[int, int], nn.Sequential] = {
            (512, 128): nn.Sequential(
                DoubleConv(512, 128),
                nn.UpsamplingBilinear2d(scale_factor=8)
            ),
            (512, 64): nn.Sequential(
                DoubleConv(512, 64),
                nn.UpsamplingBilinear2d(scale_factor=16)
            ),
            (256, 64): nn.Sequential(
                DoubleConv(256, 64),
                nn.UpsamplingBilinear2d(scale_factor=8)
            ),
        }
        self.tensor_by_channel_number: Dict[int, Tensor] = dict()
        self.channels_to_concat_by_channel: Dict[int, List[int]] = {
            512: [],
            256: [],
            128: [512],
            64: [512, 256]
        }

        '''位置编码'''
        self.pos = PositionEmbeddingLearned(512 // factor)

        '''空间注意力机制'''
        self.pam = PAMModule(512)

        '''自注意力机制'''
        self.sdpa = ScaledDotProductAttention(512)

    def forward(self, x):
        # E1: (64, 224, 224)
        x1 = self.inc(x)

        # E2: (128, 112, 112)
        x2 = self.down1(x1)

        # E3: (256, 56, 56)
        x3 = self.down2(x2)

        # E4: (512, 28, 28)
        x4 = self.down3(x3)

        # D4:(512, 14, 14)
        x5 = self.down4(x4)

        '''Setting 1'''
        x5_pam = self.pam(x5)

        '''Setting 2'''
        x5_pos = self.pos(x5)
        x5 = x5 + x5_pos

        x5_sdpa = self.sdpa(x5)
        x5 = x5_sdpa + x5_pam

        # D3: (256, 28, 28)
        x6 = self.up1(x5, x4)
        x5_scale = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_scale, x6), 1)

        # D2: (128, 56, 56)
        x7 = self.up2(x6_cat, x3)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x5_scale_f2 = F.interpolate(x5, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x5_scale_f2, x6_scale, x7), 1)

        # D1: (64, 112, 112)
        x8 = self.up3(x7_cat, x2)
        x5_scale_f8 = F.interpolate(x5, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x6_scale_f4 = F.interpolate(x6, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x6_scale_f4, x7_scale, x8), 1)

        # (128, 224, 224)
        x9 = self.up4(x8_cat, x1)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9 = torch.cat((x8_scale, x9), 1)

        logits = self.outc(x9)
        return logits
