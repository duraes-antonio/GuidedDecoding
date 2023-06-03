import random

import numpy
import torch
import torch.backends.cudnn as cudnn

from config import SEED
from model.unet_3_plus.unet_3_plus import UNet_3Plus


def load_model(load_weights=False):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = UNet_3Plus()
    # model = MTUNet(1)

    if load_weights:
        model.load_state_dict(torch.load('results/best_model.pth'))
        # model.load_state_dict(torch.load('results/best_model.pth'))

    return model


