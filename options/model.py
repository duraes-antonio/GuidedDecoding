from enum import Enum


class Models(Enum):
    MTUnet = 'mt-unet'
    MTUnetCustom = 'mt-unet-custom'
    NestedUnet = 'nested-unet'
    UNet3Plus = 'unet3+'
    TransUnet = 'trans-unet'
    TransUnetPlus = 'trans-unet++'

    def __str__(self):
        return self.value
