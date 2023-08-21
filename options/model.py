from enum import Enum


class Models(Enum):
    MTUnet = 'mt-unet'
    TransFuse = 'transfuse'
    TransFusePlusPlus = 'transfuse++'
    MTUnet3Plus = 'mt-unet3+'
    MTUnetCustom = 'mt-unet-custom'
    NestedUnet = 'nested-unet'
    UNet3Plus = 'unet3+'
    UNet3PlusCustom = 'unet3+c'
    UNetPlusPlusLuiz = 'unet++-luiz'
    UNetPlusPlus = 'unet++'
    UNetPlusPlusLB = 'unet++-lb'
    UNetPlusPlusVGG = 'unet++-vgg'
    UNetPlusPlusVGGLB = 'unet++-vgg-lb'
    UNetPlusPlusResNet = 'unet++-resnet'
    UNetPlusPlusResNetLB = 'unet++-resnet-lb'
    TransUnet = 'trans-unet'
    TransPyramidPooling = 'trans-pp'
    TransUnetPyramidPooling = 'trans-unet-pp'
    VisionTransformerAllPyramidPooling = 'trans-unet-all-pp'
    TransUnetPlus = 'trans-unet+'
    TransUnetDecoder3Skips = 'trans-unet-skips3'
    TransUnetPlusPlus = 'trans-unet++'
    TransUnetAllDecoder = 'trans-unet-ad'

    def __str__(self):
        return self.value
