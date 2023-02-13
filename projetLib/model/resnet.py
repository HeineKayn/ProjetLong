import torch.nn as nn
import torchvision

class GrayscaleResNet(torchvision.models.resnet.ResNet):

    # 224x224 par défaut
    # Output variable
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(GrayscaleResNet, self).__init__(block, layers)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

# model= GrayscaleResNet(torchvision.models.resnet.Bottleneck,[2, 2, 2, 2])
# [2, 2, 2, 2] => https://github.com/pytorch/vision/blob/791c172a337d98012018f98ffde93b1020ba3ed5/torchvision/models/resnet.py#L236