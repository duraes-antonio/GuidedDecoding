import torch

from model.AttentionUnet.attention_unet import U_Net, AttU_Net
from model.GuideDepth import GuideDepth


def load_model(model_name, weights_pth):
    model = model_builder(model_name)

    if weights_pth is not None:
        state_dict = torch.load(weights_pth, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


def model_builder(model_name):
    if model_name == 'GuideDepth':
        return GuideDepth(True)

    if model_name == 'GuideDepth-S':
        return GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])

    if model_name == 'U_Net':
        return U_Net(img_ch=3, output_ch=1)

    elif model_name == 'AttU_Net':
        return AttU_Net(img_ch=3, output_ch=1)

    print("Invalid model")
    exit(0)
