from typing import Iterable, Dict, Callable, Literal, TypedDict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.optim import Optimizer

ExecutionType = Literal["train", "test", "val"]


class CalculateMetricsParams(TypedDict):
    image: Tensor
    prediction: Tensor
    depth: Tensor
    model: nn.Module


GetMetrics = Callable[[CalculateMetricsParams], Dict[str, float]]


class LightningSegmentationRunner(LightningModule):

    def __init__(self, model: nn.Module, criterion, get_metrics: GetMetrics, checkpoint=None):
        """
        :param model: nn.Module - Modelo a ser executado
        :param criterion: loss function
        :param get_metrics: Função que calcula as métricas, recebe (input, prediction, target) e retorna um dicionário
        """
        super(LightningSegmentationRunner, self).__init__()
        self.model = model
        self.criterion = criterion
        self.get_metrics = get_metrics
        self.checkpoint = checkpoint

    def calculate_metrics(self, batch: Iterable[Tensor], execution: ExecutionType) -> Dict[str, float]:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        prediction = self.model(images)
        metrics = self.get_metrics({
            "image": images,
            "prediction": prediction,
            "depth": labels,
            "model": self.model
        })

        if execution == "test":
            return metrics

        loss: Tensor = self.criterion(prediction, labels)
        return {
            "loss": loss.item(),
            **metrics
        }

    def calculate_loss(self, batch: Iterable[Tensor]) -> Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        prediction = self.model(images)
        return self.criterion(prediction, labels)

    def training_step(self, batch: Iterable[Tensor], batch_idx: int) -> Tensor:
        values = self.calculate_metrics(batch, "train")
        values = {f'train_{k}': v for k, v in values.items()}
        self.log_dict(values, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return self.calculate_loss(batch)

    def test_step(self, batch: Iterable[Tensor], batch_idx: int) -> STEP_OUTPUT:
        values = self.calculate_metrics(batch, "test")
        values = {f'test_{k}': v for k, v in values.items()}
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)
        return values

    def validation_step(self, batch: Iterable[Tensor], batch_idx: int) -> STEP_OUTPUT:
        values = self.calculate_metrics(batch, "val")
        values = {f'val_{k}': v for k, v in values.items()}
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return self.calculate_loss(batch)

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # optimizer.load_state_dict(self.checkpoint["optimizer_states"][0])
        # self.lr_scheduler.load_state_dict(self.checkpoint["lr_scheduler"])
        return optimizer

