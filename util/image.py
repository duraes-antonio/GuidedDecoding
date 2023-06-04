import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_image_results(
        image, ground_truth, prediction, image_id,
        max_depth: float, results_path: str
):
    original_image = image[0].permute(1, 2, 0).cpu()
    ground_truth = ground_truth[0, 0].permute(0, 1).cpu()
    prediction = prediction[0, 0].permute(0, 1).detach().cpu()
    error_map = ground_truth - prediction
    max_error_value = max_depth / 10.0
    min_error_value = 0.0

    max_value = torch.max(ground_truth[ground_truth != 0.0])
    min_value = torch.min(ground_truth[ground_truth != 0.0])

    # Save original image
    save_to_dir = os.path.join(results_path, 'image_{}.png'.format(image_id))
    fig, ax = __create_fig_with_axes__()
    ax.imshow(original_image)
    fig.savefig(save_to_dir)
    plt.clf()

    # Save prediction errors
    save_to_dir = os.path.join(results_path, 'errors_{}.png'.format(image_id))
    fig, ax = __create_fig_with_axes__()
    errors = ax.imshow(error_map, vmin=min_error_value, vmax=max_error_value, cmap='Reds')
    fig.colorbar(errors, ax=ax, shrink=0.8)
    fig.savefig(save_to_dir)
    plt.clf()

    # Save ground truth
    save_to_dir = os.path.join(results_path, 'gt_{}.png'.format(image_id))
    __plot_in_file__(save_to_dir, ground_truth, min_value, max_value)

    # Save prediction depth
    save_to_dir = os.path.join(results_path, 'depth_{}.png'.format(image_id))
    __plot_in_file__(save_to_dir, prediction, min_value, max_value)


def __create_fig_with_axes__() -> Tuple[Figure, Axes]:
    figure = plt.figure(frameon=False)
    axes = plt.Axes(figure, [0., 0., 1., 1.])
    axes.set_axis_off()
    figure.add_axes(axes)
    return figure, axes


def __plot_in_file__(save_path: str, data, min_value: float, max_value: float, cmap='Greys'):
    fig, ax = __create_fig_with_axes__()
    ax.imshow(data, vmin=min_value, vmax=max_value, cmap=cmap)
    fig.savefig(save_path)
    plt.clf()
