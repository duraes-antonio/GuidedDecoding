import torch

from model.mt_unet.mt_unet import MTUNet
from model.unet_3_plus.unet_3_plus import UNet_3Plus
from options.model import Models


def load_model(model: Models, load_weights=False):
    if model == Models.UNet3Plus:
        model = UNet_3Plus()

    if model == Models.MTUnet:
        model = MTUNet(1)

    if load_weights:
        model.load_state_dict(torch.load('results/best_model.pth'))

    return model
