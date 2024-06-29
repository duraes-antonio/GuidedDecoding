import argparse
import os

from config import SEED
from dataset import datasets
from dataset.datasets import get_dataset
from evaluate import Evaluater
from options.dataset_resolution import Resolutions
from reproducibility import set_all_lib_seed
from training import Trainer


def get_args():
    file_dir = os.path.dirname(__file__)  # Directory of this path

    parser = argparse.ArgumentParser(
        description="UpSampling for Monocular Depth Estimation"
    )

    # Mode
    parser.set_defaults(train=False)
    parser.set_defaults(evaluate=False)
    parser.set_defaults(grid_search=False)
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--eval", dest="evaluate", action="store_true")
    parser.add_argument("--grid_search", dest="grid_search", action="store_true")

    # Data
    parser.add_argument(
        "--data_path", type=str, help="path to train data", default=None
    )
    parser.add_argument("--test_path", type=str, help="path to test data", default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training",
        choices=["nyu", "nyu_reduced"],
        default="nyu_reduced",
    )
    parser.add_argument(
        "--resolution",
        type=Resolutions,
        help="Resolution of the images for training",
        choices=list(Resolutions),
        default=Resolutions.Mini,
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        help="Eval mode",
        choices=["alhashim", "tu"],
        default="alhashim",
    )

    parser.add_argument(
        "--weights_path", type=str, default=None, help="path to model weights"
    )

    # Checkpoint
    parser.add_argument(
        "--load_checkpoint", type=str, help="path to checkpoint", default=""
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        help="path to save checkpoints to",
        default="./checkpoints",
    )
    parser.add_argument(
        "--save_results", type=str, help="path to save results to", default="./results"
    )

    # Optimization
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate", default=1e-4
    )
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=20)
    parser.add_argument(
        "--scheduler_step_size", type=int, help="step size of the scheduler", default=15
    )

    # System
    parser.add_argument(
        "--num_workers", type=int, help="number of dataloader workers", default=2
    )

    return parser.parse_args()


def main():
    set_all_lib_seed(SEED)
    args = get_args()
    print(args)

    if args.train:
        model_trainer = Trainer(args)
        model_trainer.train()
        args.weights_path = os.path.join(args.save_results, "best_model.pth")

    if args.evaluate:
        evaluation_module = Evaluater(args)
        evaluation_module.evaluate()

    if args.grid_search:
        dataset = get_dataset(
            args.data_path,
            split="train",
            resolution=args.resolution,
        )
        train_loader = datasets.get_dataloader(
            args.dataset,
            path=args.data_path,
            split="train",
            batch_size=args.batch_size,
            resolution=args.resolution,
            workers=args.num_workers,
        )
        # grid_search(args.data_path)


if __name__ == "__main__":
    main()
