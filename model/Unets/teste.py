# https://github.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ConvNormAct(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: int,
            norm: nn.Module = nn.BatchNorm2d,
            act: nn.Module = nn.ReLU,
            **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


class BottleNeck(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, reduction: int = 4):
        reduced_features = math.ceil(out_features / reduction)
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1X1BnReLU(in_features, reduced_features),
                        # narrow -> narrow
                        Conv3X3BnReLU(reduced_features, reduced_features),
                        # narrow -> wide
                        Conv1X1BnReLU(reduced_features, out_features, act=nn.Identity),
                    ),
                    shortcut=Conv1X1BnReLU(in_features, out_features)
                    if in_features != out_features
                    else None,
                ),
                nn.ReLU(),  # troca para leakyRelu
            )
        )


class MobileNetLikeBlock(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 2):
        # use ResidualAdd if features match, otherwise a normal Sequential
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide
                        Conv1X1BnReLU(in_features, expanded_features),
                        # wide -> wide
                        Conv3X3BnReLU(expanded_features, expanded_features),
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )


class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features,
                                      expanded_features,
                                      act=nn.ReLU6
                                      ),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )


class PEM(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(PEM, self).__init__()

        self.inter_channels = inter_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()

        # First, apply a 1x1 convolution to get inter_channels features for each spatial position
        # Resulting feature map shape: (b, inter_channels, h, w)
        inter_feats = self.conv1(x)

        # Then, apply another 1x1 convolution to project the inter_channels features
        # back to the original channel dimension
        # Resulting feature map shape: (b, out_channels, h, w)
        out = self.conv2(inter_feats)

        # Finally, calculate the pairwise dot-product between every pair of feature vectors
        # by flattening the feature map along the spatial dimensions
        # Resulting feature map shape: (b, h*w, h*w)
        out = out.view(b, self.out_channels, -1)
        out = torch.bmm(out, out.transpose(1, 2))

        # Normalize the dot-product scores with softmax along the last dimension
        # Resulting feature map shape: (b, h*w, h*w)
        out = self.softmax(out)

        # Scale the original features with the dot-product scores and add them together
        # Resulting feature map shape: (b, out_channels, h, w)
        out = torch.bmm(out, inter_feats.view(b, self.inter_channels, -1))
        out = out.view(b, self.out_channels, h, w)

        return out


class EAM(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(EAM, self).__init__()

        self.conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pem_feats):
        b, c, h, w = x.size()

        # Apply a 1x1 convolution to project the input features into the intermediate dimension
        # Resulting feature map shape: (b, inter_channels, h, w)
        proj_feats = self.conv(x)

        # Flatten the intermediate features and transpose them
        # Resulting feature map shape: (b, inter_channels, h*w)
        proj_feats = proj_feats.view(b, -1, h * w)
        proj_feats = proj_feats.transpose(1, 2)

        # Calculate the pairwise dot-product between the intermediate features
        # and the PEM features
        # Resulting feature map shape: (b, h*w, out_channels)
        weights = torch.bmm(proj_feats, pem_feats)

        # Normalize the weights with softmax along the last dimension
        # Resulting feature map shape: (b, h*w, out_channels)
        weights = self.softmax(weights)

        # Weight the PEM features with the calculated weights and sum them up
        # Resulting feature map shape: (b, out_channels, h*w)
        eam_feats = torch.bmm(weights.transpose(1, 2), pem_feats.transpose(1, 2))
        eam_feats = eam_feats.transpose(1, 2)
        eam_feats = eam_feats.view(b, -1, h, w)

        return eam_feats
