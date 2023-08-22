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
    UNetPlusPlusVGGBN = 'unet++-vgg-bn'
    UNetPlusPlusVGGBNWFU = 'unet++-vgg-bn-wfu'

    UNetPlusPlusVGGLB = 'unet++-vgg-lb'
    UNetPlusPlusResNet = 'unet++-resnet'
    UNetPlusPlusResNetWithoutFinalUp = 'unet++-resnet-wfu'
    UNetPlusPlusResNetLB = 'unet++-resnet-lb'
    UNetPlusPlusInception = 'unet++-inception-v3'
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
