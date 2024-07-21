import os
import time
from abc import ABC
from typing import Callable

import torch
import torch.optim as optim
from segmentation_models_pytorch.base import SegmentationModel
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import DEVICE, SEED
from metrics.metrics_seg import AverageMeterSegmentation
from options.dataset_resolution import Resolutions
from options.model import Models
from reproducibility import set_seed_worker


class TrainerArgs:
    def __init__(self):
        # checkpoint
        self.save_checkpoint = 'checkpoints/'
        self.save_results = 'results/'
        self.load_checkpoint = ''

        # fn(prediction: Tensor, target: Tensor) -> Tensor
        self.loss_function: Callable[[Tensor, Tensor], Tensor] = None
        self.build_model: Callable[[], SegmentationModel] = None
        self.get_dataset: Callable[[], Dataset] = None

        # data args
        self.data_path = 'data/'
        self.num_classes = 20
        self.resolution: Resolutions = Resolutions.Mini

        # model args
        self.model: Models = Models.TransUnet

        # run args
        self.batch_size = 1
        self.num_workers = 1
        self.num_epochs = 1
        self.learning_rate = 0.001
        self.scheduler_step_size = 1


def __log_metrics__(avg: AverageMeterSegmentation, average_loss: float):
    print(
        "\n*\n"
        "Average Training Loss: {average_loss:3.4f}\n"
        "ACC={average.accuracy:.3f}\n"
        "mIoU={average.mean_iou:.3f}\n"
        "Precision={average.precision:.3f}\n"
        "Recall={average.recall:.3f}\n"
        "F1 Score={average.f1_score:.3f}\n"
        "GPU Time={average.sum_gpu_time:.3f}\n"
        "Data Time={average.sum_data_time:.3f}\n"
        "*\n".format(
            average=avg, time=avg.sum_gpu_time, average_loss=average_loss
        )
    )

class Trainer(ABC):
    def __init__(self, args: TrainerArgs):
        self.device = DEVICE
        self.val_losses = []
        self.loss_func = args.loss_function

        # checkpoint
        self.checkpoint_pth = args.save_checkpoint
        self.results_pth = args.save_results

        if args.load_checkpoint != "":
            self.load_checkpoint(args.load_checkpoint)

        # model
        self.model = args.build_model()
        self.model.to(self.device)

        # data
        self.num_classes = args.num_classes

        dataset = args.get_dataset()
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(SEED)
        self.train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=set_seed_worker,
            generator=dataloader_generator,
        )

        # hyperparameters
        self.epoch = 0
        self.max_epochs = args.num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), args.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, args.scheduler_step_size, gamma=0.1
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # self.criterion = smp.losses.DiceLoss(ignore_index=255, mode='multiclass')
        # self.metrics = torchmetrics.JaccardIndex(num_classes=self.num_classes, task='multiclass')

    def train(self):
        torch.cuda.empty_cache()

        for self.epoch in range(self.epoch, self.max_epochs):
            log_step(f"\nEpoch {self.epoch} started")
            self.train_loop()
            log_step(f"Epoch {self.epoch} ended")
            self.save_checkpoint()
            self.after_epoch()

        self.after_train()
        self.save_model()

    def train_loop(self):
        self.model.train()
        accumulated_loss = 0.0

        for i, batch in enumerate(tqdm(self.train_loader)):
            # image, gt = unpack_and_move(self.device, batch)
            image: Tensor
            gt: Tensor
            image, gt = batch
            image, gt = image.to(self.device), gt.to(self.device)
            prediction = self.model(image)
            loss_value: Tensor = self.criterion(prediction, gt)

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            # accumulated_loss += loss_value.item()
            # result = SegmentationResult(self.num_classes)
            # result.evaluate(prediction.data, gt.data)
            # average_meter.update(result, gpu_time, data_time, image.size(0))

            # self.after_batch(i, image, gt, prediction)
        print(accumulated_loss)

        # Report
        time.strftime("%H:%M", time.localtime())
        average_loss = accumulated_loss / (len(self.train_loader.dataset) + 1)
        # avg = average_meter.average()
        # __log_metrics__(avg, average_loss)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epoch = checkpoint["epoch"]

    def save_checkpoint(self):
        if not os.path.isdir(self.checkpoint_pth):
            os.mkdir(self.checkpoint_pth)

        # Save checkpoint for training
        checkpoint_dir = os.path.join(
            self.checkpoint_pth, "checkpoint_{}.pth".format(self.epoch)
        )
        to_save = {
            "epoch": self.epoch + 1,
            "val_losses": self.val_losses,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(to_save, checkpoint_dir)
        log_step("Model saved")

    def save_model(self):
        if not os.path.isdir(self.results_pth):
            os.mkdir(self.results_pth)

        best_checkpoint_pth = os.path.join(
            self.checkpoint_pth, f"checkpoint_{self.max_epochs - 1}.pth"
        )
        best_model_pth = os.path.join(self.results_pth, "best_model.pth")
        checkpoint = torch.load(best_checkpoint_pth)
        torch.save(checkpoint["model"], best_model_pth)
        log_step("Model saved")

    def after_epoch(self):
        pass

    def after_batch(self, batch_index: int, image: Tensor, ground_truth: Tensor, prediction: Tensor):
        pass

    def after_train(self):
        pass


def log_step(message: str) -> None:
    current_time = time.strftime("%d/%m/%Y - %H:%M:%S", time.localtime())
    print(f'-> ({current_time}):  {message}')
