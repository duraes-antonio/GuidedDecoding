import os
import time
from typing import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torchinfo import summary
from tqdm import tqdm

from util.config import DEVICE
from dataset import datasets
from depth.losses import DepthLoss
from depth.metrics import AverageMeter, Result
from model import loader
from util.data import unpack_and_move

max_depths = {
    "nyu_reduced": 10.0,
}


class Trainer:
    def __init__(self, args):
        self.checkpoint_pth = args.save_checkpoint
        self.results_pth = args.save_results

        if not os.path.isdir(self.checkpoint_pth):
            os.mkdir(self.checkpoint_pth)

        if not os.path.isdir(self.results_pth):
            os.mkdir(self.results_pth)

        self.epoch = 0
        self.val_losses = []
        self.max_epochs = args.num_epochs
        self.maxDepth = max_depths[args.dataset]
        print("Maximum Depth of Dataset: {}".format(self.maxDepth))
        self.device = DEVICE

        # Initialize the dataset and the dataloader
        self.model: nn.Module = loader.load_model(
            args.model, args.weights_path is not None, args.weights_path, resolution=args.resolution,
            trans_unet_config=args.vit_config, num_classes=19, use_imagenet_weights=True
        )
        self.model.to(self.device)
        self.train_loader = datasets.get_dataloader(
            args.dataset,
            path=args.data_path,
            split="train",
            batch_size=args.batch_size,
            resolution=args.resolution,
            workers=args.num_workers,
        )

        if args.eval_mode == "alhashim":
            self.loss_func = DepthLoss(0.1, 1, 1, max_depth=self.maxDepth)
        else:
            self.loss_func = DepthLoss(1, 0, 0, max_depth=self.maxDepth)

        # Load Checkpoint
        if args.load_checkpoint != "":
            self.load_checkpoint(args.load_checkpoint)
            self.frozen_model()

        if args.weights_path is not None:
            self.model.segmentation_head[0] = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ).cuda()
            self.frozen_model()
            self.epoch = 0

        trainable_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        summary(model=self.model,
                input_size=(32, 3, 224, 224),  # make sure this is "input_size", not "input_shape"
                # col_names=["input_size"], # uncomment for smaller output
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        self.optimizer = optim.Adam(trainable_params, args.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, args.scheduler_step_size, gamma=0.1
        )

    def train(self):
        torch.cuda.empty_cache()

        for self.epoch in range(self.epoch, self.max_epochs):
            current_time = time.strftime("%H:%M", time.localtime())
            print("{} - Epoch {}".format(current_time, self.epoch))
            self.train_loop()
            self.save_checkpoint()

        self.save_model()

    def train_loop(self):
        self.model.train()
        accumulated_loss = 0.0
        average_meter = AverageMeter()
        # summary(self.model.cuda(), (32, 3, 256, 128))

        for i, data in enumerate(tqdm(self.train_loader)):
            t0 = time.time()
            image, gt = unpack_and_move(self.device, data)
            self.optimizer.zero_grad()
            data_time = time.time() - t0

            t0 = time.time()
            prediction = self.model(image)
            gpu_time = time.time() - t0

            loss_value = self.loss_func(prediction, gt)
            loss_value.backward()
            self.optimizer.step()

            accumulated_loss += loss_value.item()
            result = Result()
            result.evaluate(prediction.data, gt.data)
            average_meter.update(result, gpu_time, data_time, image.size(0))

        # Report
        time.strftime("%H:%M", time.localtime())
        average_loss = accumulated_loss / (len(self.train_loader.dataset) + 1)
        avg = average_meter.average()
        print(
            "\n*\n"
            "Average Training Loss: {average_loss:3.4f}\n"
            "RMSE={average.rmse:.3f}\n"
            "MAE={average.mae:.3f}\n"
            "Delta1={average.delta1:.3f}\n"
            "Delta2={average.delta2:.3f}\n"
            "Delta3={average.delta3:.3f}\n"
            "REL={average.absrel:.3f}\n"
            "Lg10={average.lg10:.3f}\n"
            "t_GPU={time:.3f}\n".format(
                average=avg, time=avg.gpu_time, average_loss=average_loss
            )
        )

    def frozen_model(self):
        encoder_modules = [
            self.model.encoder.patch_embed1,
            self.model.encoder.block1,
            self.model.encoder.norm1,

            self.model.encoder.patch_embed2,
            self.model.encoder.block2,
            self.model.encoder.norm2,

            self.model.encoder.patch_embed3,
            self.model.encoder.block3,
            self.model.encoder.norm3,

            self.model.encoder.patch_embed4,
            self.model.encoder.block4,
            self.model.encoder.norm4,
        ]
        decoder_modules = [
            self.model.decoder,
        ]
        final_head_modules = [
            self.model.segmentation_head
        ]

        to_freeze = encoder_modules[:6]

        for module in to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        l: OrderedDict = checkpoint["model"]
        l['segmentation_head.0.weight'] = l['segmentation_head.0.0.weight']
        l['segmentation_head.0.bias'] = l['segmentation_head.0.0.bias']
        del l['segmentation_head.0.0.weight']
        del l['segmentation_head.0.0.bias']

        print(checkpoint.keys())
        self.model.load_state_dict(l)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epoch = checkpoint["epoch"]
        self.model.segmentation_head[0] = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ).cuda()

    def save_checkpoint(self):
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
        current_time = time.strftime("%H:%M", time.localtime())
        print("{} - Model saved".format(current_time))

    def save_model(self):
        best_checkpoint_pth = os.path.join(
            self.checkpoint_pth, f"checkpoint_{self.max_epochs - 1}.pth"
        )
        best_model_pth = os.path.join(self.results_pth, "best_model.pth")
        checkpoint = torch.load(best_checkpoint_pth)
        torch.save(checkpoint["model"], best_model_pth)
        print("Model saved.")

    def show_images(self, image, gt, pred):
        import matplotlib.pyplot as plt

        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        gt[0, 0, gt[0, 0] == 100.0] = 0.1
        plt.imshow(image_np)
        plt.show()
        plt.imshow(gt[0, 0].cpu())
        plt.show()
        plt.imshow(pred[0, 0].detach().cpu())
        plt.show()
