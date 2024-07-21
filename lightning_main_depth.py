import math
import os
from pathlib import Path
from typing import Dict, Literal

import click
import torch
import torchvision
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

from dataset.nyu import read_nyu_csv, NyuDataset, NyuNumpyZipDataset, train_transform, test_transform

torch.set_float32_matmul_precision('medium')

from util.config import SEED
from depth.losses import DepthLoss
from model import loader
from options.dataset_resolution import Resolutions, shape_by_resolution
from options.model import Models
from segmentation.runner import LightningSegmentationRunner, CalculateMetricsParams


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


def log10(x: Tensor):
    return torch.log(x) / math.log(10)


def calculate_metrics(prediction: Tensor, depth: Tensor) -> Dict[str, float]:
    abs_diff = (prediction - depth).abs()
    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    mae = float(abs_diff.mean())
    l1_fn = nn.L1Loss()
    l1 = float(l1_fn(prediction, depth))
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
    depth = params['depth']
    return calculate_metrics(prediction, depth)


def get_calc_test_metrics_fn(resolution: Resolutions):
    def inverse_depth_norm(depth: Tensor, max_depth=10.0):
        depth = max_depth / depth
        depth = torch.clamp(depth, max_depth / 100, max_depth)
        return depth

    def calculate_test_metrics(params: CalculateMetricsParams) -> Dict[str, float]:
        prediction = params['prediction']
        gt = params['depth']
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


RunModes = ['train', 'test']


@click.command()
@click.option("-e", "--max_epochs", type=click.INT, default=20)
@click.option("-b", "--batch_size", type=click.INT, default=16)
@click.option("-w", "--num_workers", type=click.INT, default=4)
@click.option("-m", "--mode", type=click.Choice(RunModes), default='train')
@click.option(
    "-s",
    "--size",
    type=click.Choice(list(map(str, Resolutions))),
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
    default="checkpoints/checkpoints_depth/",
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
    default="models_depth/",
)
@click.option(
    "-tdp",
    "--train_data_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    default="data/nyu2_train.csv",
)
@click.option(
    "-tedp",
    "--test_data_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    default="NYU_Testset.zip",
)
def main(
        max_epochs: int,
        batch_size: int,
        num_workers: int,
        size: Resolutions,
        checkpoint_load_path: Path,
        checkpoint_save_path: Path,
        model_save_path: Path,
        train_data_path: Path,
        test_data_path: Path,
        mode: Literal['train', 'test']
):
    seed_everything(seed=SEED, workers=True)
    percent_to_train = 90
    max_depth = 10.0

    if not os.path.isdir(checkpoint_save_path):
        os.mkdir(checkpoint_save_path)

    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    log_dir = f'results_depth_logs/'
    loggers = [TensorBoardLogger(log_dir)]

    size = Resolutions(size)
    final_size = shape_by_resolution[size]
    loss_function = DepthLoss(0.1, 1, 1, max_depth=max_depth)

    if mode == 'train':
        train_paths = read_nyu_csv(str(train_data_path))
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
        # model = loader.load_model(
        #     Models.UNetMixedTransformer, False, resolution=size,
        #     num_classes=1,
        # )

        # Carregar model treinado em segmentação
        model = loader.load_model(
            Models.UNetMixedTransformer, False, resolution=size,
            num_classes=19,
        )
        checkpoint = torch.load(checkpoint_load_path, map_location=torch.device('cuda'))
        model_state = checkpoint['state_dict']
        model_state = {k.replace("model.", ""): v for k, v in model_state.items()}
        # model.load_state_dict(model_state)
        encoder_weights = {
            k.replace("encoder.", ""): v
            for k, v in model_state.items()
            if k.startswith("encoder.")
        }
        decoder_weights = {
            k.replace("decoder.", ""): v
            for k, v in model_state.items()
            if k.startswith("decoder.")
        }
        model.encoder.load_state_dict(encoder_weights)
        model.decoder.load_state_dict(decoder_weights)

        lightning_train_model = LightningSegmentationRunner(
            model=model,
            criterion=loss_function,
            get_metrics=calculate_train_metrics,
            checkpoint=checkpoint,
        )
        model.segmentation_head[0] = nn.Sequential(
            nn.Conv2d(16, 1, (3, 3), (1, 1), (1, 1)),
        ).cuda()

        randint = str(torch.randint(0, 1000, (1,)).item())
        callbacks = [
            # FinetuningScheduler(ft_schedule="results_depth_logs/lightning_logs/version_45"
            #                                 "/LightningSegmentationRunner_ft_schedule_test.yaml"),
            EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=True),
            ModelCheckpoint(
                monitor='train_loss',
                dirpath='models_depth/',
                filename='_'.join([f'size-{size}', '{epoch}-epochs', 'data-aug', randint]),
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
        print('_'.join([f'size-{size}', '{epoch}-epochs', 'data-aug', randint]))

        transform_test = test_transform(final_size)
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
            model=model, criterion=loss_function, get_metrics=get_calc_test_metrics_fn(size), checkpoint=checkpoint
        )
        test_trainer = Trainer(logger=loggers, enable_checkpointing=False)
        test_trainer.test(model=runner, dataloaders=test_loader)

    if mode == 'test':
        transform_test = test_transform(final_size)
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

        # Carregar model treinado em segmentação
        model = loader.load_model(
            Models.UNetMixedTransformer, False, resolution=size,
            num_classes=19,
        )
        checkpoint = torch.load(checkpoint_load_path, map_location=torch.device('cuda'))

        print(checkpoint['state_dict'].keys())
        model_state = checkpoint['state_dict']
        model_state = {k.replace("model.", ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)

        # encoder_weights = {
        #     k: v for k, v in model_state.items() if
        #     k.startswith("encoder.")
        # }
        # decoder_weights = {k: v for k, v in model_state.items() if k.startswith("decoder.")}
        # model.encoder.load_state_dict(encoder_weights)
        # model.decoder.load_state_dict(decoder_weights)

        model.segmentation_head[0] = nn.Sequential(
            nn.Conv2d(16, 1, (3, 3), (1, 1), (1, 1)),
        ).cuda()

        runner: LightningSegmentationRunner = LightningSegmentationRunner(
            model=model, criterion=loss_function, get_metrics=get_calc_test_metrics_fn(size), checkpoint=checkpoint
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

    return None


if __name__ == "__main__":
    main()
