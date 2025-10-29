import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

def _cifar_stem(m: ResNet):
    # Za CIFAR-10 koristimo 3x3 conv i izbacujemo maxpool jer su slike male (32x32)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

class ResNetCIFAR(ResNet):
    def __init__(self, block, layers, num_classes=10):
        super().__init__(block, layers, num_classes=num_classes)
        _cifar_stem(self)

def resnet18_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50_cifar(num_classes=10):
    return ResNetCIFAR(Bottleneck, [3, 4, 6, 3], num_classes)
