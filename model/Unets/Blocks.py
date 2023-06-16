import torch.nn.functional as F

from .teste import *


class FSA(nn.Module):  # Full Scale Aggregation
    """
    O primeiro passo é interpolar bilinearmente cada mapa de características para o mesmo tamanho do maior mapa.
      Por exemplo, se F1 tem 256x256 pixels e F2 tem 128x128 pixels, você precisa interpolar F2 para 256x256 pixels
      usando uma função de interpolação bilinear.
    O segundo passo é concatenar todos os mapas de características interpolados ao longo do eixo dos canais.
      Por exemplo, se cada mapa tem 64 canais, você obtém um tensor concatenado com 256 canais.
    O terceiro passo é aplicar uma convolução 1x1 ao tensor concatenado para reduzir o número de canais
      para o mesmo número do mapa original. Por exemplo, se você quer obter um mapa unificado com 64 canais,
      você usa uma convolução 1x1 com 64 filtros.
    """

    def __init__(self, in_channels_list, out_channels):
        super(FSA, self).__init__()
        self.fuse_layer = nn.Conv2d(sum(in_channels_list), out_channels, kernel_size=1)

    def forward(self, x_list, target):
        target_size = target.size()[2:]
        interpolated_tensors = []

        # step 1
        for i in range(len(x_list)):
            if x_list[i].size()[2:] != target_size:  # check if the tensor size is different from the target size
                interpolated_tensor = F.interpolate(x_list[i], size=target_size, mode='bilinear', align_corners=True)
            else:
                interpolated_tensor = x_list[i]  # no interpolation needed
            interpolated_tensors.append(interpolated_tensor)

        # step 2
        concatenated_tensor = torch.cat(interpolated_tensors, dim=1)

        # i think, i can put a custom convolution block here..

        # step 3
        output_tensor = self.fuse_layer(concatenated_tensor)

        return output_tensor


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),  # nn.Hardswish() é oque o mobile usa
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.convblock(x)


class ConvBlock_BottleNeck(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_BottleNeck, self).__init__()
        self.convblock = nn.Sequential(
            MobileNetLikeBlock(in_channels, out_channels),
            MobileNetLikeBlock(out_channels, out_channels)
            # nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.convblock(x)


# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         return out

class Up_concat(nn.Module):
    # upscale e convBlock

    def __init__(self, in_channels=None):
        super().__init__()

        # self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2) # sobe a resolução
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input, concat_with):
        up = self.up(input)
        inter = F.interpolate(up, size=[concat_with[0].size(2), concat_with[0].size(3)], mode='bilinear',
                              align_corners=True)
        concat_with.append(inter)
        concat = torch.cat(concat_with, dim=1)
        return concat
