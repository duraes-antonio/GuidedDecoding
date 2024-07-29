"""
Code from FastDepth 
    Diana Wofk et al, FastDepth: Fast Monocular Depth
    Estimation on Embedded Devices, International Conference on Robotics and 
    Automation (ICRA), 2019
    https://github.com/dwofk/fast-depth
"""
import math
from typing import Dict

import numpy as np
import torch
import torchvision
from torch import Tensor

from options.dataset_resolution import Resolutions, shape_by_resolution
from segmentation.runner import CalculateMetricsParams, GetMetrics


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x."""
    return torch.log(x) / math.log(10)


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.rmse_log = 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.rmse_log = np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(
            self,
            irmse,
            imae,
            mse,
            rmse,
            mae,
            rmse_log,
            absrel,
            lg10,
            delta1,
            delta2,
            delta3,
            gpu_time,
            data_time,
    ):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.rmse_log = rmse_log
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.rmse_log = math.sqrt(torch.pow(log10(output) - log10(target), 2).mean())
        self.absrel = float((abs_diff / target).mean())

        max_ratio = torch.max(output / target, target / output)
        self.delta1 = float((max_ratio < 1.25).float().mean())
        self.delta2 = float((max_ratio < 1.25 ** 2).float().mean())
        self.delta3 = float((max_ratio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.count = None
        self.sum_irmse = None
        self.sum_imae = None
        self.sum_mse = None
        self.sum_rmse = None
        self.sum_mae = None
        self.sum_rmse_log = None
        self.sum_absrel = None
        self.sum_lg10 = None
        self.sum_delta1 = None
        self.sum_delta2 = None
        self.sum_delta3 = None
        self.sum_data_time = None
        self.sum_gpu_time = None
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_rmse_log = 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_rmse_log += n * result.rmse_log
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count,
            self.sum_imae / self.count,
            self.sum_mse / self.count,
            self.sum_rmse / self.count,
            self.sum_mae / self.count,
            self.sum_rmse_log / self.count,
            self.sum_absrel / self.count,
            self.sum_lg10 / self.count,
            self.sum_delta1 / self.count,
            self.sum_delta2 / self.count,
            self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count,
            self.sum_data_time / self.count,
        )
        return avg


def calculate_metrics(prediction: Tensor, depth: Tensor) -> Dict[str, float]:
    abs_diff = (prediction - depth).abs()
    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    mae = float(abs_diff.mean())
    lg10 = float((log10(prediction) - log10(depth)).abs().mean())
    absrel = float((abs_diff / depth).mean())

    max_ratio = torch.max(prediction / depth, depth / prediction)
    delta1 = float((max_ratio < 1.25).float().mean())
    delta2 = float((max_ratio < 1.25 ** 2).float().mean())
    delta3 = float((max_ratio < 1.25 ** 3).float().mean())

    return {
        "rmse": rmse,
        "mae": mae,
        "absrel": absrel,
        "lg10": lg10,
        "d1": delta1,
        "d2": delta2,
        "d3": delta3,
    }


def calculate_train_metrics(params: CalculateMetricsParams) -> Dict[str, float]:
    prediction = params['prediction']
    depth = params['ground_truth']
    return calculate_metrics(prediction, depth)


def get_calc_test_metrics_fn(resolution: Resolutions) -> GetMetrics:
    def inverse_depth_norm(depth: Tensor, max_depth=10.0):
        depth = max_depth / depth
        depth = torch.clamp(depth, max_depth / 100, max_depth)
        return depth

    def calculate_test_metrics(params: CalculateMetricsParams) -> Dict[str, float]:
        prediction = params['prediction']
        gt = params['ground_truth']
        image = params['image']
        model = params['model']
        crop = [20, 460, 24, 616]
        size = shape_by_resolution[resolution]
        resizer = torchvision.transforms.Resize(size)

        gt_flip = torch.flip(gt, [3])
        image_flip = torch.flip(image, [3])
        image_flip = resizer(image_flip)

        prediction = inverse_depth_norm(prediction)
        prediction_flip = inverse_depth_norm(model(image_flip))

        resize_to_gt_size = torchvision.transforms.Resize(gt.shape[-2:])
        prediction = resize_to_gt_size(prediction)
        prediction_flip = resize_to_gt_size(prediction_flip)

        gt = gt[:, :, crop[0]: crop[1], crop[2]: crop[3]]
        gt_flip = gt_flip[:, :, crop[0]: crop[1], crop[2]: crop[3]]
        prediction = prediction[:, :, crop[0]: crop[1], crop[2]: crop[3]]
        prediction_flip = prediction_flip[:, :, crop[0]: crop[1], crop[2]: crop[3]]

        result = calculate_metrics(prediction, gt)
        result_flip = calculate_metrics(prediction_flip, gt_flip)
        metrics = result.keys()
        return {metric: (result[metric] + result_flip[metric]) / 2 for metric in metrics}

    return calculate_test_metrics
