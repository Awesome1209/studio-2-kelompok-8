import timm
from torchvision import transforms

NAME = "convnext_tiny"
IMG = 256


def get_model():
    return timm.create_model(NAME, pretrained=False, num_classes=5)


def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG / 0.875)),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
