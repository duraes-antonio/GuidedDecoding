from typing import Optional, OrderedDict

import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import SegmentationModel
from torch.nn import ELU

from model.mt_unet.mt_unet import MTUNet
from model.mt_unet.mt_unet_3_plus import MTUNet3Plus
from model.trans_att_unet.TransAttUnet import UNet_Attention_Transformer_Multiscale
from model.trans_unet.vit_seg_modeling import TransUnetConfigType, CONFIGS, VisionTransformer
from model.unet_3_plus.unet_3_plus import UNet_3Plus
from model.unet_plus_plus.nested import NestedUNet, NestedUNetLuizBlock
from model.unet_plus_plus.nested_luiz import LuizNestedUNet
from model.unet_plus_plus.nested_resnet import NestedUNetResNet2, NestedUNetResNet, NestedUNetResNetLB
from model.unet_plus_plus.nested_vgg import NestedUNetVGG16BN_ECH, NestedUNetVGG16BN_ECG_BL, NestedUNetVGG16BN_EL_BL, \
    NestedUNetVGG16BNWFU_ECH, NestedUNetVGGLE, NestedUNetVGGLuizBlock
from options.dataset_resolution import Resolutions, shape_by_resolution
from options.model import Models



def load_model(
        model: Models, load_weights=False,
        path_weights='results/best_model.pth',
        resolution=Resolutions.Half,
        trans_unet_config: TransUnetConfigType = TransUnetConfigType.r50_vit_b16,
        num_classes=1, eval=False, useImageNetWeights=True
):
    model_instance = get_segmentation_models(model, num_classes, useImageNetWeights)

    if model == Models.UNet3Plus:
        model_instance = UNet_3Plus()

    if model == Models.UNetPlusPlusLuiz:
        model_instance = LuizNestedUNet()

    if model == Models.UNetPlusPlus:
        model_instance = NestedUNet()

    if model == Models.UNetPlusPlus_BL:
        model_instance = NestedUNetLuizBlock()

    """VGG16 Batch normalization"""
    if model == Models.UNetPlusPlusVGGBN:
        model_instance = NestedUNetVGG16BN_ECH()

    if model == Models.UNetPlusPlusVGGBN_BL:
        model_instance = NestedUNetVGG16BN_ECG_BL()

    if model == Models.UNetPlusPlusVGGBN_EL_BL:
        model_instance = NestedUNetVGG16BN_EL_BL()

    if model == Models.UNetPlusPlusVGGBN_WFU:
        model_instance = NestedUNetVGG16BNWFU_ECH()

    if model == Models.UNetPlusPlusVGG:
        model_instance = NestedUNetVGGLE()

    if model == Models.UNetPlusPlusVGG_BL:
        model_instance = NestedUNetVGGLuizBlock()

    """ResNet 101"""
    if model == Models.UNetPlusPlusResNet_WFU:
        model_instance = NestedUNetResNet2()

    if model == Models.UNetPlusPlusResNet:
        model_instance = NestedUNetResNet()

    if model == Models.UNetPlusPlusResNet_BL:
        model_instance = NestedUNetResNetLB()

    if model == Models.MTUnet:
        model_instance = MTUNet(1)

    if model == Models.MTUnet3Plus:
        model_instance = MTUNet3Plus(1)

    if model == Models.TransAttentionUnet:
        model_instance = UNet_Attention_Transformer_Multiscale(3, 1)

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


    if eval and load_weights:
        l: OrderedDict = torch.load(path_weights)
        l['segmentation_head.0.weight'] = l['segmentation_head.0.0.weight']
        l['segmentation_head.0.bias'] = l['segmentation_head.0.0.bias']
        del l['segmentation_head.0.0.weight']
        del l['segmentation_head.0.0.bias']
        model_instance.load_state_dict(l)
        print("EVAL and Weights loaded from: ", path_weights)

    elif load_weights:
        model_instance.load_state_dict(torch.load(path_weights))
        print("Weights loaded from: ", path_weights)

    return model_instance

def get_segmentation_models(model: Models, num_classes=1, load_weight=True) -> Optional[SegmentationModel]:
    sm_default_args = {"encoder_weights": "imagenet" if load_weight else None, "classes": num_classes, "activation": ELU}
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
