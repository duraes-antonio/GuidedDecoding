from typing import Optional

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import SegmentationModel
from torch.nn import ELU

from model.aa_b4.aa_b4 import FCB4
from options.model import Models


def load_model(load_weights=False, path_weights="results/best_model.pth"):
    model_instance = FCB4()

    if load_weights:
        model_instance.load_state_dict(torch.load(path_weights))

    return model_instance


def get_segmentation_models(model: Models) -> Optional[SegmentationModel]:
    sm_default_args = {"encoder_weights": "imagenet", "classes": 1, "activation": ELU}
    unet_plus_plus_encoder_by_enum = {
        Models.UNetPlusPlusSENet154: "senet154",
        Models.UNetPlusPlusInceptionResNetv2: "inceptionresnetv2",
        Models.UNetPlusPlusVGG19BN: "vgg19_bn",
        Models.UNetPlusPlusXception: "tu-xception71",
        Models.UNetPlusPlusEfficientNet: "efficientnet-b6",
    }

    if model in unet_plus_plus_encoder_by_enum:
        return smp.UnetPlusPlus(
            encoder_name=unet_plus_plus_encoder_by_enum[model], **sm_default_args
        )

    unet_encoder_by_enum = {
        Models.UNetSENet154: "senet154",
        Models.UNetInceptionResNetv2: "inceptionresnetv2",
        Models.UNetVGG19BN: "vgg19_bn",
        Models.UNetXception: "tu-xception71",
        Models.UNetMixedTransformer: "mit_b2",
        Models.UNetEfficientNet: "efficientnet-b6",
    }

    if model in unet_encoder_by_enum:
        return smp.Unet(encoder_name=unet_encoder_by_enum[model], **sm_default_args)

    unet_3_plus_encoder_by_enum = {
        Models.UNet3PlusSENet154: "senet154",
        Models.UNet3PlusInceptionResNetv2: "inceptionresnetv2",
        Models.UNet3PlusVGG19BN: "vgg19_bn",
        Models.UNet3PlusXception: "tu-xception71",
        Models.UNet3PlusMixedTransformer: "mit_b2",
        Models.UNet3PlusEfficientNet: "efficientnet-b6",
    }

    if model in unet_3_plus_encoder_by_enum:
        return smp.Unet3Plus(
            encoder_name=unet_3_plus_encoder_by_enum[model], **sm_default_args
        )

    ma_net_encoder_by_enum = {
        Models.MANetSENet154: "senet154",
        Models.MANetInceptionResNetV2: "inceptionresnetv2",
        Models.MANetXception: "tu-xception71",
        Models.MANetMixedTransformer: "mit_b2",
    }

    if model in ma_net_encoder_by_enum:
        return smp.MAnet(encoder_name=ma_net_encoder_by_enum[model], **sm_default_args)

    pan_net_encoder_by_enum = {
        Models.PANNetSENet154: "senet154",
        Models.PANNetXception: "tu-xception71",
    }

    if model in pan_net_encoder_by_enum:
        return smp.PAN(encoder_name=pan_net_encoder_by_enum[model], **sm_default_args)

    return None
