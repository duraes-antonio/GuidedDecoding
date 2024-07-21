import os
from pathlib import Path

import click
import torch
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils import data
from torch.utils.data import DataLoader

from config import SEED
from model import loader
from options.dataset_resolution import Resolutions
from options.model import Models
from segmentation.dataset import CityscapesDataset
from train_segmentation.loss import cross_entropy2d
from train_segmentation.metrics import get_metrics_lightning
from segmentation.runner import LightningSegmentationRunner

torch.set_float32_matmul_precision('medium')


def save_checkpoint(epoch, model, optimizer, lr_scheduler, checkpoint_pth):
    if not os.path.isdir(checkpoint_pth):
        os.mkdir(checkpoint_pth)

    # Save checkpoint for training
    checkpoint_dir = os.path.join(
        checkpoint_pth, "checkpoint_{}.pth".format(epoch)
    )
    to_save = {
        "epoch": epoch + 1,
        "val_losses": 1,
        "model": model.cuda().state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(to_save, checkpoint_dir)


def save_model(max_epochs, checkpoint_pth, results_pth):
    if not os.path.isdir(results_pth):
        os.mkdir(results_pth)

    best_checkpoint_pth = os.path.join(
        checkpoint_pth, f"checkpoint_{max_epochs - 1}.pth"
    )
    best_model_pth = os.path.join(results_pth, "best_model.pth")
    checkpoint = torch.load(best_checkpoint_pth)
    torch.save(checkpoint["model"], best_model_pth)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., prob=0.5):
        self.std = std
        self.mean = mean
        self.prob = prob

    def __call__(self, tensor):
        if torch.rand(1) < self.prob:
            return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


factors = ['1', '2', '4', '8']


@click.command()
@click.option("-e", "--max_epochs", type=click.INT, default=100)
@click.option("-b", "--batch_size", type=click.INT, default=32)
@click.option("-w", "--num_workers", type=click.INT, default=4)
@click.option(
    "-f",
    "--factor",
    type=click.Choice(factors),
    default=Resolutions.Mini.value
)
@click.option(
    "-clp",
    "--checkpoint_load_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    default=None
)
@click.option(
    "-csp",
    "--checkpoint_save_path",
    type=click.Path(
        exists=False,
        file_okay=False,
        readable=False,
        path_type=Path,
    ),
    default="checkpoints_segmentation/",
)
@click.option(
    "-msp",
    "--model_save_path",
    type=click.Path(
        exists=False,
        file_okay=False,
        readable=False,
        path_type=Path,
    ),
    default="models_segmentation/",
)
@click.option(
    "-tdp",
    "--train_data_path",
    type=click.Path(
        exists=True,
        file_okay=False,
        readable=True,
        path_type=Path,
    ),
    default="data",
)
def main(
        max_epochs: int,
        batch_size: int,
        num_workers: int,
        factor: int,
        checkpoint_load_path: Path,
        checkpoint_save_path: Path,
        model_save_path: Path,
        train_data_path: Path,
):
    print(f"Running with parameters: {max_epochs=}, {batch_size=}, {num_workers=}, {factor=}")

    path_data = "data/"
    factor = int(factor)
    seed_everything(seed=SEED, workers=True)
    img_size = (512 // factor, 1024 // factor)
    percent_to_train = 90

    flip_prob = 0.25
    import torchvision.transforms.v2 as transforms
    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(p=flip_prob),
            # transforms.RandomRotation(degrees=25),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
            # AddGaussianNoise(mean=0., std=1., prob=0.15),
            transforms.ToTensor(),
        ],
    )
    train_dataset = CityscapesDataset(
        root=path_data,
        split='train',
        img_size=img_size,
        transforms=transform
    )
    test_dataset = CityscapesDataset(
        root=path_data,
        split='val',
        img_size=img_size
    )
    train_set_size = int(len(train_dataset) * percent_to_train / 100)
    valid_set_size = len(train_dataset) - train_set_size

    seed = torch.Generator().manual_seed(SEED)
    train_subset, valid_subset = data.random_split(
        train_dataset,
        [train_set_size, valid_set_size],
        generator=seed
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    model = loader.load_model(
        Models.UNetMixedTransformer, False, resolution=Resolutions.Full,
        num_classes=19,
    )

    lightning_model = LightningSegmentationRunner(
        model=model,
        criterion=cross_entropy2d,
        get_metrics=get_metrics_lightning
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=True),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='models_segmentation/',
            filename='_'.join([f'factor-{factor}', '{epoch}-epochs', 'cb']),
            mode='min',
        ),
        # FinetuningScheduler()
    ]

    log_dir = f'results_segmentation_logs/'
    loggers = [
        TensorBoardLogger(log_dir)
    ]
    train_trainer = Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        log_every_n_steps=5,
        logger=loggers,
    )
    train_trainer.fit(lightning_model, train_loader, valid_loader)

    test_trainer = Trainer(logger=loggers)
    test_trainer.test(model=lightning_model, dataloaders=test_loader)
    return None


if __name__ == "__main__":
    main()
