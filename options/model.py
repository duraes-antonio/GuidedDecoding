import ssl
from enum import Enum

ssl._create_default_https_context = ssl._create_unverified_context


class Models(Enum):
    TransUnet = 'trans-unet'

    UNetPlusPlusInceptionResNetv2 = 'unet++_inception-resnet-v2'
    UNetPlusPlusVGG19BN = 'unet++_vgg-19-bn'
    UNetPlusPlusXception = 'unet++_xception'

    UNetInceptionResNetv2 = 'unet_inception-resnet-v2'
    UNetVGG19BN = 'unet_vgg-19-bn'
    UNetXception = 'unet_xception'
    UNetMixedTransformer = 'unet_mt'

    def __str__(self):
        return self.value
