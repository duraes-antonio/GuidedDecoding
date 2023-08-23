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
    UNetPlusPlus_BL = 'unet++-bl'

    UNetPlusPlusVGG = 'unet++-vgg'

    UNetPlusPlusVGGBN = 'unet++-vgg-bn'
    UNetPlusPlusVGGBN_WFU = 'unet++-vgg-bn-wfu'
    UNetPlusPlusVGGBN_BL = 'unet++-vgg-bn-bl'
    UNetPlusPlusVGGBN_EL_BL = 'unet++-vgg-bn-el-bl'

    UNetPlusPlusVGG_BL = 'unet++-vgg-bl'
    UNetPlusPlusResNet = 'unet++-resnet'
    UNetPlusPlusResNet_WFU = 'unet++-resnet-wfu'
    UNetPlusPlusResNet_BL = 'unet++-resnet-bl'
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
