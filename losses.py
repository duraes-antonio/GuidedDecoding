""" 
Depth Loss by Alhashim et al.:

Ibraheem Alhashim, High Quality Monocular Depth Estimation via
Transfer Learning, https://arxiv.org/abs/1812.11941, 2018

https://github.com/ialhashim/DenseDepth
"""

from math import exp

import torch
from torch.nn import functional


def gaussian(window_size: int, sigma: float):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def gradient(x):
    """
    idea from tf.image.image_gradients(image)
    https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    """
    left = x
    right = functional.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = functional.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top

    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def gradient_loss(gen_frames, gt_frames, alpha=1):
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)

    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    grad_comb = grad_diff_x ** alpha + grad_diff_y ** alpha

    return torch.mean(grad_comb)


class DepthLoss:
    def __init__(self, alpha: float, beta: float, gamma: float, max_depth=10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.maxDepth = max_depth
        self.L1_Loss = torch.nn.L1Loss()

    def __call__(self, output, depth):
        if self.beta == 0 and self.gamma == 0:
            valid_mask = depth > 0.0
            output = output[valid_mask]
            depth = depth[valid_mask]
            l_depth = self.L1_Loss(output, depth)
            loss = l_depth
        else:
            l_depth = self.L1_Loss(output, depth)
            l_ssim = torch.clamp((1 - self.ssim(output, depth, self.maxDepth)) * 0.5, 0, 1)
            l_grad = gradient_loss(output, depth)

            loss = self.alpha * l_depth + self.beta * l_ssim + self.gamma * l_grad
        return loss

    @staticmethod
    def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
        L = val_range
        padding = 0
        (_, channel, height, width) = img1.size()

        if window is None:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel=channel).to(img1.device)
            padding = window_size // 2

        mu1 = functional.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = functional.conv2d(img2, window, padding=padding, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = functional.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = functional.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = functional.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

        c1 = (0.01 * L) ** 2
        c2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + c2
        v2 = sigma1_sq + sigma2_sq + c2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret
