import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm

from train_segmentation.metrics import get_metrics


def validate(val_loader: DataLoader, model: nn.Module, device):
    # tldr: to make layers behave differently during inference (vs training)
    model.eval()

    # enable calculation of confusion matrix for n_classes = 19
    # running_metrics_val = runningScore(19)

    # empty list to add Accuracy and Jaccard Score Calculations
    acc_sh = []
    js_sh = []
    f1_sh = []
    recall_sh = []
    precision_sh = []

    accuracy_arr = []
    iou_arr = []
    f1_arr = []
    recall_arr = []
    precision_arr = []

    with torch.no_grad():
        for image_num, (val_images, val_labels) in tqdm(enumerate(val_loader)):
            val_images: Tensor = val_images.to(device)
            val_labels: Tensor = val_labels.to(device)

            # Model prediction
            val_pred: Tensor = model(val_images)

            # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
            # considering predictions with highest scores for each pixel among 19 classes
            pred = val_pred.data.max(1)[1].cpu().numpy()
            gt = val_labels.data.cpu().numpy()

            # Updating Mertics
            # running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics['accuracy'])
            js_sh.append(sh_metrics['iou'])
            f1_sh.append(sh_metrics['f1'])
            recall_sh.append(sh_metrics['recall'])
            precision_sh.append(sh_metrics['precision'])

            accuracy_arr.append(sh_metrics['Accuracy'])
            iou_arr.append(sh_metrics['Iou'])
            f1_arr.append(sh_metrics['F1'])
            recall_arr.append(sh_metrics['Recall'])
            precision_arr.append(sh_metrics['Precision'])

            # miou_arr.append(miou(val_pred, val_labels))

    #            # break for testing purpose
    #             if image_num == 10:
    #                 break

    # score = running_metrics_val.get_scores()
    score = {}
    # running_metrics_val.reset()

    acc_s = sum(acc_sh) / len(acc_sh)
    js_s = sum(js_sh) / len(js_sh)
    f1_s = sum(f1_sh) / len(f1_sh)
    recall_s = sum(recall_sh) / len(recall_sh)
    precision_s = sum(precision_sh) / len(precision_sh)

    Accuracy_s = sum(accuracy_arr) / len(accuracy_arr)
    Iou_s = sum(iou_arr) / len(iou_arr)
    F1_s = sum(f1_arr) / len(f1_arr)
    Recall_s = sum(recall_arr) / len(recall_arr)
    Precision_s = sum(precision_arr) / len(precision_arr)

    # miou_s = sum(miou_arr)/len(miou_arr)
    score["acc"] = acc_s
    score["iou"] = js_s
    score["f1"] = f1_s
    score["recall"] = recall_s
    score["precision"] = precision_s

    score["Accuracy"] = Accuracy_s
    score["Iou"] = Iou_s
    score["F1"] = F1_s
    score["Recall"] = Recall_s
    score["Precision"] = Precision_s

    print("Different Metrics were: ", score)
    return score