import torch.nn.functional as F
from torch import Tensor


# Cross Entropy Loss adapted from meetshah1995 to prevent size inconsistencies between model precition
# and target label
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/loss.py

def cross_entropy2d(image: Tensor, target: Tensor, weight=None, size_average=True):
    n, c, h, w = image.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        image = F.interpolate(image, size=(ht, wt), mode="bilinear", align_corners=True)

    image = image.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        image, target, weight=weight, ignore_index=250,
        reduction='mean'
    )
    return loss
