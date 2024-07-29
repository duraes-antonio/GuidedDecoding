from os import path
from pathlib import Path
from random import shuffle
from typing import List, Callable

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, random_split

from dataset.nyu import test_transform, NyuNumpyZipDataset, read_nyu_csv, train_transform, NyuDataset
from depth.metrics import get_calc_test_metrics_fn, calculate_train_metrics
from options.dataset_resolution import Resolutions, shape_by_resolution
from segmentation.runner import LightningSegmentationRunner
from util.config import SEED


def run_test(
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loggers: List[Logger],
        test_data_path: Path,
        resolution: Resolutions,
):
    model.eval()
    size = shape_by_resolution[resolution]
    transform_test = test_transform(size)
    test_dataset = NyuNumpyZipDataset(
        zip_path=str(test_data_path),
        transform=transform_test,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )
    runner: LightningSegmentationRunner = LightningSegmentationRunner(
        model=model, get_metrics=get_calc_test_metrics_fn(resolution),
        optimizer=optimizer, lr_scheduler=scheduler
    )
    # ev = Evaluater(
    #     resolution=size,
    #     model=model.model,
    #     dataset_path=str(test_data_path),
    #     num_workers=num_workers,
    # )
    # ev.evaluate()
    test_trainer = Trainer(logger=loggers, enable_checkpointing=False)
    test_trainer.test(model=runner, dataloaders=test_loader)


def run_train(
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_function: Callable,
        loggers: List[Logger],
        train_data_path: Path,
        resolution: Resolutions,
        batch_size: int,
        num_workers: int,
        max_epochs: int,
        percent_to_train: int,
        percent_dataset_used: int,
        checkpoint_dir: str,
        checkpoint_filename: str,
):
    final_size = shape_by_resolution[resolution]
    train_paths = read_nyu_csv(str(train_data_path))
    shuffle(train_paths)
    train_paths = train_paths[:int(len(train_paths) * percent_dataset_used / 100)]
    transform_train = train_transform(final_size)
    train_dataset = NyuDataset(
        pairs_path=train_paths,
        transform=transform_train,
        split='train',
    )

    train_val_set_size = len(train_dataset)
    train_set_size = int(train_val_set_size * percent_to_train / 100)
    valid_set_size = train_val_set_size - train_set_size

    seed = torch.Generator().manual_seed(SEED)
    train_subset, valid_subset = random_split(
        train_dataset,
        [train_set_size, valid_set_size],
        generator=seed
    )
    train_loader = DataLoader(
        train_dataset,
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
    lightning_train_model = LightningSegmentationRunner(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        criterion=loss_function,
        get_metrics=calculate_train_metrics,
    )
    print(f'\n--> Path to save checkpoint: {path.join(checkpoint_dir, checkpoint_filename)}\n')
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=True),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=checkpoint_dir,
            filename=checkpoint_filename,
            mode='min',
        )
    ]
    train_trainer = Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        log_every_n_steps=5,
        logger=loggers,
    )
    train_trainer.fit(lightning_train_model, train_loader, valid_loader)
