import os
import time

import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import datasets
from losses import Depth_Loss
from metrics import AverageMeter, Result
from model import loader

max_depths = {
    'nyu_reduced': 10.0,
}


class Trainer():
    def __init__(self, args):
        self.debug = True

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
        print('Maximum Depth of Dataset: {}'.format(self.maxDepth))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize the dataset and the dataloader
        self.model = loader.load_model()
        self.model.to(self.device)
        self.train_loader = datasets.get_dataloader(args.dataset,
                                                    path=args.data_path,
                                                    split='train',
                                                    augmentation=args.eval_mode,
                                                    batch_size=args.batch_size,
                                                    resolution=args.resolution,
                                                    workers=args.num_workers)
        self.optimizer = optim.Adam(self.model.parameters(), args.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, args.scheduler_step_size, gamma=0.1
        )

        if args.eval_mode == 'alhashim':
            self.loss_func = Depth_Loss(0.1, 1, 1, maxDepth=self.maxDepth)
        else:
            self.loss_func = Depth_Loss(1, 0, 0, maxDepth=self.maxDepth)

        # Load Checkpoint
        if args.load_checkpoint != '':
            self.load_checkpoint(args.load_checkpoint)

    def train(self):
        torch.cuda.empty_cache()
        self.start_time = time.time()

        for self.epoch in range(self.epoch, self.max_epochs):
            current_time = time.strftime('%H:%M', time.localtime())
            print('{} - Epoch {}'.format(current_time, self.epoch))
            self.train_loop()
            self.save_checkpoint()

        self.save_model()

    def train_loop(self):
        self.model.train()
        accumulated_loss = 0.0
        average_meter = AverageMeter()

        for i, data in enumerate(tqdm(self.train_loader)):
            t0 = time.time()
            image, gt = self.unpack_and_move(data)
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
        time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (len(self.train_loader.dataset) + 1)
        avg = average_meter.average()
        print(
            '\n*\n'
            'Average Training Loss: {average_loss:3.4f}\n'
            'RMSE={average.rmse:.3f}\n'
            'MAE={average.mae:.3f}\n'
            'Delta1={average.delta1:.3f}\n'
            'Delta2={average.delta2:.3f}\n'
            'Delta3={average.delta3:.3f}\n'
            'REL={average.absrel:.3f}\n'
            'Lg10={average.lg10:.3f}\n'
            't_GPU={time:.3f}\n'.format(average=avg, time=avg.gpu_time, average_loss=average_loss)
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch']

    def save_checkpoint(self):
        # Save checkpoint for training
        checkpoint_dir = os.path.join(self.checkpoint_pth, 'checkpoint_{}.pth'.format(self.epoch))
        torch.save({
            'epoch': self.epoch + 1,
            'val_losses': self.val_losses,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }, checkpoint_dir)
        current_time = time.strftime('%H:%M', time.localtime())
        print('{} - Model saved'.format(current_time))

    def save_model(self):
        best_checkpoint_pth = os.path.join(self.checkpoint_pth, f'checkpoint_{self.max_epochs - 1}.pth')
        best_model_pth = os.path.join(self.results_pth, 'best_model.pth')
        checkpoint = torch.load(best_checkpoint_pth)
        torch.save(checkpoint['model'], best_model_pth)
        print('Model saved.')

    def inverse_depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth[zero_mask] = 0.0
        return depth

    def depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        depth[zero_mask] = 0.0
        return depth

    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):
            keys = data.keys()
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt
        print('Type not supported')

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
