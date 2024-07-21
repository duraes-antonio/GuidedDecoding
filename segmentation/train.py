from typing import List

from segmentation_models_pytorch.losses import DiceLoss
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from train_segmentation.loss import cross_entropy2d

criteria = cross_entropy2d


def train(train_loader: DataLoader, model: nn.Module, optimizer: Optimizer, device) -> List[float]:
    count = 0

    # List to cumulate loss during iterations
    loss_list = []
    for (images, labels) in train_loader:
        count += 1

        # we used model.eval() below. This is to bring model back to training mood.
        model.train()

        images = images.to(device)
        labels = labels.to(device)

        # Model Prediction
        pred = model(images)

        # Loss Calculation
        loss = criteria(pred, labels)
        loss_list.append(loss)

        # optimiser
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (loss_list)
