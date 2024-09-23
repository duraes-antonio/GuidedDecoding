from torch import nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    encoder_children = encoder.children()

    for child in encoder_children:
        for param in child.parameters():
            param.requires_grad = False

    return encoder


def unfreeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    return
