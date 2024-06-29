import os
import time

import torch
import torchvision
from tqdm import tqdm

from config import DEVICE
from dataset import datasets
from dataset import transforms
from metrics import AverageMeter, Result
from model import loader
from options.dataset_resolution import shape_by_resolution
from util.data import unpack_and_move
from util.image import save_image_results
from util.log import print_metrics

max_depths = {
    "nyu": 10.0,
    "nyu_reduced": 10.0,
}

resolutions = {
    "nyu": shape_by_resolution,
    "nyu_reduced": shape_by_resolution,
}

crops = {"nyu": [20, 460, 24, 616], "nyu_reduced": [20, 460, 24, 616]}


class Evaluater:
    def __init__(self, args):
        self.debug = True
        self.dataset = args.dataset

        self.maxDepth = max_depths[args.dataset]
        self.res_dict = resolutions[args.dataset]
        self.resolution = self.res_dict[args.resolution]
        print("Resolution for Eval: {}".format(self.resolution))
        self.resolution_keyword = args.resolution
        print("Maximum Depth of Dataset: {}".format(self.maxDepth))
        self.crop = crops[args.dataset]
        self.eval_mode = args.eval_mode
        self.result_dir = args.save_results

        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.device = DEVICE
        self.model = loader.load_model(True, args.weights_path)
        self.model.to(self.device)
        self.test_loader = datasets.get_dataloader(
            args.dataset,
            path=args.test_path,
            split="test",
            batch_size=1,
            resolution=args.resolution,
            workers=args.num_workers,
        )

        self.downscale_image = torchvision.transforms.Resize(
            self.resolution
        )  # To Model resolution
        self.to_tensor = transforms.ToTensor(test=True, max_depth=self.maxDepth)
        self.visualize_images = [
            0,
            1,
            2,
            3,
            4,
            5,
            100,
            101,
            102,
            103,
            104,
            105,
            200,
            201,
            202,
            203,
            204,
            205,
            300,
            301,
            302,
            303,
            304,
            305,
            400,
            401,
            402,
            403,
            404,
            405,
            500,
            501,
            502,
            503,
            504,
            505,
            600,
            601,
            602,
            603,
            604,
            605,
        ]

    def evaluate(self):
        self.model.eval()
        average_meter = AverageMeter()

        for i, data in enumerate(tqdm(self.test_loader)):
            t0 = time.time()
            image, gt = data
            packed_data = {"image": image[0], "depth": gt[0]}
            data = self.to_tensor(packed_data)
            image, gt = unpack_and_move(self.device, data)
            image = image.unsqueeze(0)
            gt = gt.unsqueeze(0)

            image_flip = torch.flip(image, [3])
            gt_flip = torch.flip(gt, [3])
            if self.eval_mode == "alhashim":
                # For model input
                image = self.downscale_image(image)
                image_flip = self.downscale_image(image_flip)

            data_time = time.time() - t0

            t0 = time.time()

            inv_prediction = self.model(image)
            prediction = self.inverse_depth_norm(inv_prediction)

            inv_prediction_flip = self.model(image_flip)
            prediction_flip = self.inverse_depth_norm(inv_prediction_flip)

            gpu_time = time.time() - t0

            if self.eval_mode == "alhashim":
                upscale_depth = torchvision.transforms.Resize(
                    gt.shape[-2:]
                )  # To GT res

                prediction = upscale_depth(prediction)
                prediction_flip = upscale_depth(prediction_flip)

                if i in self.visualize_images:
                    save_image_results(
                        image, gt, prediction, i, self.maxDepth, self.result_dir
                    )

                gt = gt[:, :, self.crop[0]: self.crop[1], self.crop[2]: self.crop[3]]
                gt_flip = gt_flip[
                          :, :, self.crop[0]: self.crop[1], self.crop[2]: self.crop[3]
                          ]
                prediction = prediction[
                             :, :, self.crop[0]: self.crop[1], self.crop[2]: self.crop[3]
                             ]
                prediction_flip = prediction_flip[
                                  :, :, self.crop[0]: self.crop[1], self.crop[2]: self.crop[3]
                                  ]

            result = Result()
            result.evaluate(prediction.data, gt.data)
            average_meter.update(result, gpu_time, data_time, image.size(0))

            result_flip = Result()
            result_flip.evaluate(prediction_flip.data, gt_flip.data)
            average_meter.update(result_flip, gpu_time, data_time, image.size(0))

        # Report
        avg = average_meter.average()
        current_time = time.strftime("%H:%M", time.localtime())
        self.save_results(avg)
        print_metrics(avg)

    def save_results(self, average):
        results_file = os.path.join(self.result_dir, "results.txt")
        with open(results_file, "w") as f:
            f.write("RMSE,MAE,REL, RMSE_log,Lg10,Delta1,Delta2,Delta3\n")
            f.write(
                "{average.rmse:.3f}"
                ",{average.mae:.3f}"
                ",{average.absrel:.3f}"
                ",{average.rmse_log:.3f}"
                ",{average.lg10:.3f}"
                ",{average.delta1:.3f}"
                ",{average.delta2:.3f}"
                ",{average.delta3:.3f}".format(average=average)
            )

    def inverse_depth_norm(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        return depth

    def depth_norm(self, depth):
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        return depth
