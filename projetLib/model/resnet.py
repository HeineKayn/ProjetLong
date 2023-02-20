import torch.nn as nn
import torchvision

class GrayscaleResNet(torchvision.models.resnet.ResNet):

    # 224x224 par défaut
    # Output variable
    def __init__(self, block, layers, num_classes=1):
        self.inplanes = 64
        super(GrayscaleResNet, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

# model= GrayscaleResNet(torchvision.models.resnet.Bottleneck,[3, 4, 6, 3])
# [3, 4, 6, 3] => https://github.com/pytorch/vision/blob/791c172a337d98012018f98ffde93b1020ba3ed5/torchvision/models/resnet.py#L236