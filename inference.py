import argparse
import os
import time

import tensorrt as trt
import torch
import torchvision
from torch2trt import torch2trt

from util.config import DEVICE
from data import transforms
from dataset.datasets import get_dataloader
from metrics import AverageMeter, Result
from model import loader
from options.dataset_resolution import shape_by_resolution, Resolutions
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


def get_args():
    parser = argparse.ArgumentParser(
        description="Nano Inference for Monocular Depth Estimation"
    )

    # Mode
    parser.set_defaults(evaluate=False)
    parser.add_argument("--eval", dest="evaluate", action="store_true")

    # Data
    parser.add_argument("--test_path", type=str, help="path to test data")
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
        choices=[Resolutions.Full, Resolutions.Half],
        default=Resolutions.Half,
    )

    # Model
    parser.add_argument(
        "--model", type=str, help="name of the model to be trained", default="UpDepth"
    )
    parser.add_argument("--weights_path", type=str, help="path to model weights")
    parser.add_argument(
        "--save_results", type=str, help="path to save results to", default="./results"
    )

    # System
    parser.add_argument(
        "--num_workers", type=int, help="number of dataloader workers", default=1
    )

    return parser.parse_args()


class Inference_Engine:
    def __init__(self, args):
        args.resolution: Resolutions
        self.maxDepth = max_depths[args.dataset]
        self.res_dict = resolutions[args.dataset]
        self.resolution = self.res_dict[args.resolution]
        self.resolution_keyword = args.resolution.value
        print("Resolution for Eval: {}".format(self.resolution))
        print("Maximum Depth of Dataset: {}".format(self.maxDepth))
        self.crop = crops[args.dataset]

        self.result_dir = args.save_results
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.results_filename = "{}_{}_{}".format(
            args.dataset, args.resolution, args.model
        )

        self.device = DEVICE
        self.model = loader.load_model(args.weights_path)
        self.model = self.model.eval().cuda()

        if args.evaluate:
            self.test_loader = get_dataloader(
                args.dataset,
                path=args.test_path,
                split="test",
                batch_size=1,
                resolution=args.resolution,
                uncompressed=True,
                workers=args.num_workers,
            )

        if args.resolution == Resolutions.Half:
            self.upscale_depth = torchvision.transforms.Resize(
                self.res_dict["full"]
            )  # To Full res
            self.downscale_image = torchvision.transforms.Resize(
                self.resolution
            )  # To Half res

        self.to_tensor = transforms.ToTensor(test=True, max_depth=self.maxDepth)

        self.visualize_images = []

        self.trt_model, _ = self.convert_PyTorch_to_TensorRT()

        self.run_evaluation()

    def run_evaluation(self):
        speed_pyTorch = self.pyTorch_speedtest()
        speed_tensorRT = self.tensorRT_speedtest()
        average = self.tensorRT_evaluate()
        self.save_results(average, speed_tensorRT, speed_pyTorch)

    def pyTorch_speedtest(self, num_test_runs=200):
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            result = self.model(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print("[PyTorch] Runtime: {}s".format(times))
        print("[PyTorch] FPS: {}\n".format(fps))
        return times

    def tensorRT_speedtest(self, num_test_runs=200):
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            result = self.trt_model(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print("[tensorRT] Runtime: {}s".format(times))
        print("[tensorRT] FPS: {}\n".format(fps))
        return times

    def convert_PyTorch_to_TensorRT(self):
        x = torch.ones([1, 3, *self.resolution]).cuda()
        print("[tensorRT] Starting TensorRT conversion")
        model_trt = torch2trt(self.model, [x], fp16_mode=True)
        print("[tensorRT] Model converted to TensorRT")

        TRT_LOGGER = trt.Logger()
        file_path = os.path.join(
            self.result_dir, "{}.engine".format(self.results_filename)
        )
        with open(file_path, "wb") as f:
            f.write(model_trt.engine.serialize())

        with open(file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        print("[tensorRT] Engine serialized\n")
        return model_trt, engine

    def tensorRT_evaluate(self):
        torch.cuda.empty_cache()
        self.model = None
        average_meter = AverageMeter()

        dataset = self.test_loader.dataset
        for i, data in enumerate(dataset):
            t0 = time.time()
            image, gt = data
            packed_data = {"image": image, "depth": gt}
            data = self.to_tensor(packed_data)
            image, gt = unpack_and_move(self.device, data)
            image = image.unsqueeze(0)
            gt = gt.unsqueeze(0)

            image_flip = torch.flip(image, [3])
            gt_flip = torch.flip(gt, [3])
            if self.resolution_keyword == Resolutions.Half.value:
                image = self.downscale_image(image)
                image_flip = self.downscale_image(image_flip)

            torch.cuda.synchronize()
            data_time = time.time() - t0

            t0 = time.time()
            inv_prediction = self.trt_model(image)
            prediction = self.inverse_depth_norm(inv_prediction)
            torch.cuda.synchronize()
            gpu_time0 = time.time() - t0

            t1 = time.time()
            inv_prediction_flip = self.trt_model(image_flip)
            prediction_flip = self.inverse_depth_norm(inv_prediction_flip)
            torch.cuda.synchronize()
            gpu_time1 = time.time() - t1

            if self.resolution_keyword == Resolutions.Half.value:
                prediction = self.upscale_depth(prediction)
                prediction_flip = self.upscale_depth(prediction_flip)

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
            average_meter.update(result, gpu_time0, data_time, image.size(0))

            result_flip = Result()
            result_flip.evaluate(prediction_flip.data, gt_flip.data)
            average_meter.update(result_flip, gpu_time1, data_time, image.size(0))

        # Report
        avg = average_meter.average()
        current_time = time.strftime("%H:%M", time.localtime())
        print_metrics(avg)
        return avg

    def save_results(self, average, trt_speed, pyTorch_speed):
        file_path = os.path.join(
            self.result_dir, "{}.txt".format(self.results_filename)
        )
        with open(file_path, "w") as f:
            f.write("s[PyTorch], s[tensorRT], RMSE,MAE,REL,Lg10,Delta1,Delta2,Delta3\n")
            f.write(
                "{pyTorch_speed:.3f}"
                ",{trt_speed:.3f}"
                ",{average.rmse:.3f}"
                ",{average.mae:.3f}"
                ",{average.absrel:.3f}"
                ",{average.lg10:.3f}"
                ",{average.delta1:.3f}"
                ",{average.delta2:.3f}"
                ",{average.delta3:.3f}".format(
                    average=average, trt_speed=trt_speed, pyTorch_speed=pyTorch_speed
                )
            )

    def inverse_depth_norm(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        return depth

    def depth_norm(self, depth):
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        return depth


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


if __name__ == "__main__":
    args = get_args()
    print(args)

    engine = Inference_Engine(args)
