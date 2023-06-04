from enum import Enum


class Models(Enum):
    MTUnet = 'mt-unet'
    UNet3Plus = 'unet3+'

    def __str__(self):
        return self.value
