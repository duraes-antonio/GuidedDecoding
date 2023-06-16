import torch.nn as nn
from torchvision import models


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
    print("Parametros do Encoder: %.2f M" % count)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        backbone_nn = models.efficientnet_b0(pretrained=True)

        print("NOT freezing backbone layers - ", type(backbone_nn).__name__)
        for param in backbone_nn.parameters():
            param.requires_grad = True

        count_parameters(backbone_nn)
        self.original_model = backbone_nn

    # def forward(self, x):
    #     features = [x]
    #     # Process the first convolutional layer and max pooling
    #     x = self.original_model.conv1(x)
    #     x = self.original_model.maxpool(x)
    #     features.append(x)

    #     # Process the stages (ShuffleNetV2 layers)
    #     for stage in self.original_model.stage2, self.original_model.stage3, self.original_model.stage4:
    #         for layer in stage:
    #             x = layer(x)
    #             features.append(x)

    #     # Process the last convolutional layer
    #     x = self.original_model.conv5(x)
    #     features.append(x)

    #     # if True: # leitura de tamanho das features
    #     #     for block in range(len(features)):
    #     #         print("feature[{}]: {}".format(block,features[block].size()))

    #     return features

    def forward(self, x):
        features = [x]
        for _, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))

        # if True: # leitura de tamanho das features
        #     for block in range(len(features)):
        #         print("feature[{}]: {}".format(block,features[block].size()))

        return features

# shufflenet_v2_x1_0 - 2.28 M > 3.77 M

# shufflenet_v2_x0_5 - 1.37 M > 2.86 M

# efficientNet b0 - 5.29M > 6.78M


# squeezenet1_0 - 1.25M > 2.74M (foi pra 28.53M ++ e 113.81M o classico)
#                        bz, ch, he, wi
# feature[0]: torch.Size([1, 3, 240, 320])-
# feature[1]: torch.Size([1, 96, 117, 157])
# feature[2]: torch.Size([1, 96, 117, 157])-
# feature[3]: torch.Size([1, 96, 58, 78])
# feature[4]: torch.Size([1, 128, 58, 78])
# feature[5]: torch.Size([1, 128, 58, 78])
# feature[6]: torch.Size([1, 256, 58, 78]) -
# feature[7]: torch.Size([1, 256, 29, 39])
# feature[8]: torch.Size([1, 256, 29, 39])
# feature[9]: torch.Size([1, 384, 29, 39]) -
# feature[10]: torch.Size([1, 384, 29, 39])
# feature[11]: torch.Size([1, 512, 29, 39])-
# feature[12]: torch.Size([1, 512, 14, 19])
# feature[13]: torch.Size([1, 512, 14, 19])-
