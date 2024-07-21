from torchvision.transforms import ToTensor, Compose, Resize, Normalize

transform = Compose(
    [
        Resize((256, 512)),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(),
    ]
)
