import torch.nn as nn
from torchvision import models


class LuizInceptionEncoder(nn.Module):
    def __init__(self):
        super(LuizInceptionEncoder, self).__init__()
        self.inception_model = models.inception_v3(pretrained=True)
        self.encoder = nn.Sequential(*list(self.inception_model.children())[:-2])

    def forward(self, x):
        enc_features = [x]

        for layer in self.encoder:
            x = layer(x)
            enc_features.append(x)
        return enc_features
