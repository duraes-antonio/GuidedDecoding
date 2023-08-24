import ssl
from enum import Enum

ssl._create_default_https_context = ssl._create_unverified_context


class Models(Enum):
    Custom = 'custom'
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
    TransUnet = 'trans-unet'

    UNetPlusPlusSENet154 = 'unet++_senet-154'
    UNetPlusPlusInceptionResNetv2 = 'unet++_inception-resnet-v2'
    UNetPlusPlusVGG19BN = 'unet++_vgg-19-bn'
    UNetPlusPlusXception = 'unet++_xception'

    UNetSENet154 = 'unet_senet-154'
    UNetInceptionResNetv2 = 'unet_inception-resnet-v2'
    UNetVGG19BN = 'unet_vgg-19-bn'
    UNetXception = 'unet_xception'
    UNetMixedTransformer = 'unet_mt'

    MANetSENet154 = 'ma-net_senet-154'
    MANetInceptionResNetv2 = 'ma-net_inception-resnet-v2'
    MANetXception = 'ma-net_xception'
    MANetMixedTransformer = 'ma-net_mt'

    PANNetSENet154 = 'pan_senet-154'
    PANNetVGG19BN = 'pan_vgg-19-bn'
    PANNetXception = 'pan_xception'

    def __str__(self):
        return self.value
