import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision lama pakai ini:
# from torchvision.models.utils import load_state_dict_from_url
# sekarang ganti ke:
from torch.hub import load_state_dict_from_url
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True, num_classes=1000):
        super(ResNet, self).__init__()

        # pilih backbone
        if arch == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet152":
            backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError("Unknown arch: %s" % arch)

        # ambil semua layer kecuali fc
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(pretrained=True, num_classes=1000):
    return ResNet(arch="resnet50", pretrained=pretrained, num_classes=num_classes)


def resnet101(pretrained=True, num_classes=1000):
    return ResNet(arch="resnet101", pretrained=pretrained, num_classes=num_classes)


def resnet152(pretrained=True, num_classes=1000):
    return ResNet(arch="resnet152", pretrained=pretrained, num_classes=num_classes)