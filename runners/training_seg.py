import matplotlib.pyplot as plt
from torch import Tensor

from runners.training import Trainer, TrainerArgs


def __show_images__(image: Tensor, gt: Tensor, prediction: Tensor) -> None:
    image_np = image[0].cpu().permute(1, 2, 0).numpy()
    gt[0, 0, gt[0, 0] == 100.0] = 0.1

    # show the image
    plt.imshow(image_np)
    plt.show()

    # show the ground truth
    plt.imshow(gt[0, 0].cpu())
    plt.show()

    # show the prediction
    plt.imshow(prediction[0, 0].detach().cpu())
    plt.show()


class SegmentationTrainer(Trainer):
    def __init__(self, args: TrainerArgs):
        super().__init__(args)


    def after_batch(self, batch_index: int, image: Tensor, ground_truth: Tensor, prediction: Tensor):
        __show_images__(image, ground_truth, prediction)
