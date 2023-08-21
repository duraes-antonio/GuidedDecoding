import numpy as np
import torch

from model.mt_unet.mt_unet import MTUNet
from model.mt_unet.mt_unet_3_plus import MTUNet3Plus
from model.trans_fuse.TransFuse import TransFuse_S
from model.trans_fuse.TransFuse_pp import TransFuse_S_PlusPlus
from model.trans_pp.vit_seg_modeling import TransPyramidPooling
from model.trans_unet.vit_seg_modeling import VisionTransformer, CONFIGS, TransUnetConfigType
from model.trans_unet_all_pp.vit_seg_modeling import VisionTransformerAllPyramidPooling
from model.trans_unet_partial_pp.vit_seg_modeling import VisionTransformerPyramidPooling
from model.unet_3_plus.unet_3_plus import UNet_3Plus
from model.unet_3_plus_custom_encoder.unet_3_plus import Unet3PlusCustomEncoder
from model.unet_plus_plus.nested import NestedUNet, NestedUNetLuizBlock
from model.unet_plus_plus.nested_luiz import LuizNestedUNet
from model.unet_plus_plus.nested_resnet import NestedUNetResNetLE
from model.unet_plus_plus.nested_vgg import NestedUNetVGGLE, NestedUNetVGGLuizBlock
from options.dataset_resolution import Resolutions, shape_by_resolution
from options.model import Models


def load_model(
        model: Models, load_weights=False,
        path_weights='results/best_model.pth',
        resolution=Resolutions.Half,
        trans_unet_config: TransUnetConfigType = TransUnetConfigType.r50_vit_b16
):
    if model == Models.UNet3Plus:
        model = UNet_3Plus()

    if model == Models.UNet3PlusCustom:
        model = Unet3PlusCustomEncoder()

    if model == Models.UNetPlusPlusLuiz:
        model = LuizNestedUNet()

    if model == Models.UNetPlusPlus:
        model = NestedUNet()

    if model == Models.UNetPlusPlusLB:
        model = NestedUNetLuizBlock()

    if model == Models.UNetPlusPlusVGG:
        model = NestedUNetVGGLE()

    if model == Models.UNetPlusPlusVGGLB:
        model = NestedUNetVGGLuizBlock()

    if model == Models.UNetPlusPlusResNet:
        model = NestedUNetResNetLE()

    if model == Models.MTUnet:
        model = MTUNet(1)

    if model == Models.TransFuse:
        model = TransFuse_S(1, pretrained=True)

    if model == Models.TransFusePlusPlus:
        model = TransFuse_S_PlusPlus(1, pretrained=True)

    if model == Models.MTUnet3Plus:
        model = MTUNet3Plus(1)

    if model == Models.TransUnet:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))

    if model == Models.TransPyramidPooling:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = TransPyramidPooling(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
        print('\nTransPyramidPooling\n')

    if model == Models.TransUnetPyramidPooling:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformerPyramidPooling(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
        print('\nVisionTransformerPyramidPooling\n')

    if model == Models.VisionTransformerAllPyramidPooling:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformerAllPyramidPooling(config_vit, img_size=img_size,
                                                   num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
        print('\nVisionTransformerAllPyramidPooling\n')

    if load_weights:
        model.load_state_dict(torch.load(path_weights))
    print(type(model))
    return model
