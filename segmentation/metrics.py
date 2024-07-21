from typing import Dict

from numpy import ndarray
import numpy as np
import sklearn.metrics as skm
from torch import Tensor

from segmentation.runner import CalculateMetricsParams


def get_metrics(ground_truth: Tensor, prediction: Tensor) -> Dict[str, float]:
    # confusion_matrix: ndarray = skm.multilabel_confusion_matrix(ground_truth, prediction)
    # tn = confusion_matrix[:, 0, 0]
    # tp = confusion_matrix[:, 1, 1]
    # fn = confusion_matrix[:, 1, 0]
    # fp = confusion_matrix[:, 0, 1]
    pred_numpy = prediction.data.max(1)[1].cpu().numpy().flatten()
    gt_numpy = ground_truth.data.cpu().numpy().flatten()

    acc = skm.accuracy_score(gt_numpy, pred_numpy)
    iou = skm.jaccard_score(gt_numpy, pred_numpy, average='micro')
    f1 = skm.f1_score(gt_numpy, pred_numpy, average='micro')
    precision = skm.precision_score(gt_numpy, pred_numpy, average='micro')
    recall = skm.recall_score(gt_numpy, pred_numpy, average='micro')

    result_gm_sh = {
        'accuracy': acc,
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
    return result_gm_sh

def get_metrics_lightning(params: CalculateMetricsParams) -> Dict[str, float]:
    return get_metrics(params['depth'], params['prediction'])