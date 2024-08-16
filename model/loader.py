from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel
from torch.nn import ELU

from model.trans_unet.vit_seg_modeling import TransUnetConfigType, CONFIGS, VisionTransformer
from options.dataset_resolution import Resolutions, shape_by_resolution
from options.model import Models


def load_model(
        model: Models,
        resolution=Resolutions.Half,
        trans_unet_config: TransUnetConfigType = TransUnetConfigType.r50_vit_b16,
        num_classes=1
):
    model_instance = get_segmentation_models(model, num_classes)

    if model == Models.TransUnet:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model_instance = VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        model_instance.load_from(weights=np.load(config_vit.pretrained_path))

    return model_instance


def get_segmentation_models(model: Models, num_classes=1, load_weight=True) -> Optional[SegmentationModel]:
    sm_default_args = {"encoder_weights": "imagenet" if load_weight else None, "classes": num_classes,
                       "activation": ELU}
    unet_plus_plus_encoder_by_enum = {
        Models.UNetPlusPlusInceptionResNetv2: "inceptionresnetv2",
        Models.UNetPlusPlusVGG19BN: "vgg19_bn",
        Models.UNetPlusPlusXception: "tu-xception71",
    }

    if model in unet_plus_plus_encoder_by_enum:
        return smp.UnetPlusPlus(
            encoder_name=unet_plus_plus_encoder_by_enum[model], **sm_default_args
        )

    unet_encoder_by_enum = {
        Models.UNetInceptionResNetv2: "inceptionresnetv2",
        Models.UNetVGG19BN: "vgg19_bn",
        Models.UNetXception: "xception",
        Models.UNetMixedTransformerB2: "mit_b2",
        Models.UNetMixedTransformerB3: "mit_b3",
        Models.UNetMixedTransformerB4: "mit_b4",
    }

    if model in unet_encoder_by_enum:
        return smp.Unet(encoder_name=unet_encoder_by_enum[model], **sm_default_args)

    return None
