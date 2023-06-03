import torch

from model.unet_3_plus.unet_3_plus import UNet_3Plus


def load_model(load_weights=False):

    model = UNet_3Plus()
    # model = MTUNet(1)

    if load_weights:
        model.load_state_dict(torch.load('results/best_model.pth'))
        # model.load_state_dict(torch.load('results/best_model.pth'))

    return model


