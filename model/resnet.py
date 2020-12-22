import torch
from torchvision import models


class resnet(torch.nn.Module):

    def __init__(self, net="resnet18"):
        super().__init__()
        if net == "resnet18":
            self.features = models.resnet18()
        if net == "resnet34":
            self.features = models.resnet34()
        if net == "resnet50":
            self.features = models.resnet50()
        if net == "resnet101":
            self.features = models.resnet101()
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model = resnet('resnet50')
    print(model)
    x = torch.rand(2, 3, 256, 256)
    y = model(x)
