import numpy as np
import torch

from model.Unets.NestedUnet import NestedUNet
from model.mt_unet.mt_unet import MTUNet
from model.trans_unet.vit_seg_modeling import VisionTransformer, CONFIGS, TransUnetConfigType
from model.trans_unet_3_skips.vit_seg_modeling import VisionTransformerSkips3
from model.trans_unet_plus_plus.vit_seg_modeling import VisionTransformer2
from model.unet_3_plus.unet_3_plus import UNet_3Plus
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

    if model == Models.MTUnet:
        model = MTUNet(1)

    if model == Models.NestedUnet:
        model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False).cuda()

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

    if model == Models.TransUnetDecoder3Skips:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformerSkips3(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
        print('\nVisionTransformer(Skip 3)\n')

    if model == Models.TransUnetAllDecoder:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformer2(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
        print('\nVisionTransformer (All Decoder)\n')

    if load_weights:
        model.load_state_dict(torch.load(path_weights))

    return model
