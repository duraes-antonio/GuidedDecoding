import os
from pathlib import Path
from typing import Optional, Literal

import click
import torch
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.data import DataLoader

from lightning_main_depth import RunModes
from model import loader
from options.dataset_resolution import Resolutions
from options.model import Models
from segmentation.dataset import CityscapesDataset
from segmentation.loss import cross_entropy2d
from segmentation.metrics import get_metrics_lightning
from segmentation.runner import LightningSegmentationRunner
from util.config import SEED

torch.set_float32_matmul_precision('medium')


def save_checkpoint(epoch, model, optimizer, lr_scheduler, checkpoint_pth, filename: str):
    if not os.path.isdir(checkpoint_pth):
        os.mkdir(checkpoint_pth)

    # Save checkpoint for training
    checkpoint_dir = os.path.join(
        checkpoint_pth, filename or "checkpoint_{}.pth".format(epoch)
    )
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(to_save, checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir)
    torch.save(checkpoint["model"], checkpoint_dir.replace('.pth', '_model.pth'))


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
        readable=True,
        path_type=Path,
    ),
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
@click.option(
    "-mf",
    "--model_filename",
    type=click.STRING,
)
@click.option(
    "-du",
    "--dataset_usage",
    type=click.IntRange(1, 100),
    default=100
)
@click.option("-m", "--mode", type=click.Choice(RunModes), default='train')
def main(
        max_epochs: int,
        batch_size: int,
        num_workers: int,
        factor: int,
        checkpoint_load_path: Path,
        checkpoint_save_path: Path,
        model_save_path: Path,
        train_data_path: Path,
        model_filename: Optional[str],
        mode: Literal['train', 'test'],
        dataset_usage: int = 100
):
    print(f"Running with parameters: {max_epochs=}, {batch_size=}, {num_workers=}, {factor=}")

    if not model_save_path:
        raise ValueError("Model save path (-msp, --model_save_path) must be provided")

    path_data = "data/"
    factor = int(factor)
    seed_everything(seed=SEED, workers=True)
    img_size = (512 // factor, 1024 // factor)
    percent_to_train = 90

    import torchvision.transforms.v2 as transforms
    transform = transforms.Compose(
        [
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
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
        Models.UNetMixedTransformer, resolution=Resolutions.Full,
        num_classes=19,
    )
    # model_loader = ModelLoader(
    #     target_model=Models.UNetMixedTransformer,
    #     resolution=Resolutions.Full,
    #     checkpoint_load_path=str(checkpoint_load_path),
    #     from_task=Task.SEGMENTATION,
    #     to_task=Task.SEGMENTATION
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 15, 0.1)
    lightning_model = LightningSegmentationRunner(
        model=model,
        criterion=cross_entropy2d,
        get_metrics=get_metrics_lightning,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )

    optimizer_used = len(optimizer.state) > 0
    scheduler_used = scheduler.last_epoch > 0

    model_filename_params = [
        f'factor-{factor}',
        f'ds-usage-{dataset_usage}',
        f'batch-{batch_size}',
        f'opt-{int(optimizer_used)}',
        f'scheduler-{int(scheduler_used)}',
    ]

    if checkpoint_load_path is None:
        model_filename_params = ['base'] + model_filename_params

    log_root_path = os.path.join('segmentation', 'logs', model_save_path)
    log_last_dir = '_'.join(model_filename_params)
    loggers = [TensorBoardLogger(log_root_path, name=log_last_dir)]

    auto_model_filename = '_'.join(model_filename_params + ['{epoch}'])
    model_filename = model_filename or auto_model_filename
    model_save_path = os.path.join('segmentation', model_save_path)

    if mode == 'train':
        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=True),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=model_save_path,
                filename=model_filename,
                mode='min',
            ),
        ]
        train_trainer = Trainer(
            callbacks=callbacks,
            max_epochs=max_epochs,
            log_every_n_steps=5,
            logger=loggers,
        )
        train_trainer.fit(lightning_model, train_loader, valid_loader)
        save_checkpoint(max_epochs, model, optimizer, scheduler, model_save_path, model_filename + '.pth')

    test_lightning_model = LightningSegmentationRunner(
        model=model,
        criterion=cross_entropy2d,
        get_metrics=get_metrics_lightning,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )
    test_trainer = Trainer(logger=loggers)
    test_trainer.test(model=test_lightning_model, dataloaders=test_loader)

    return None


if __name__ == "__main__":
    main()
