from enum import Enum


class Models(Enum):
    MTUnet = 'mt-unet'
    MTUnetCustom = 'mt-unet-custom'
    NestedUnet = 'nested-unet'
    UNet3Plus = 'unet3+'
    TransUnet = 'trans-unet'
    TransFuse = 'trans-fuse'
    TransUnetPlus = 'trans-unet+'
    TransUnetDecoder3Skips = 'trans-unet-skips3'
    TransUnetPlusPlus = 'trans-unet++'
    TransUnetAllDecoder = 'trans-unet-ad'

    def __str__(self):
        return self.value
