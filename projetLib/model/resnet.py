import torch.nn as nn
import torchvision

class GrayscaleResNet(torchvision.models.resnet.ResNet):

    # 224x224 par défaut
    # Output variable
    def __init__(self, block, layers,channels=1, num_classes=1):
        self.inplanes = 64
        super(GrayscaleResNet, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

def getCNNresnet(x,channels):
    resnets = {
        50 : [3, 4, 6, 3],
        101 : [3, 4, 23, 3],
        152 : [3, 8, 36, 3]
    }
    basicResnet = GrayscaleResNet(torchvision.models.resnet.Bottleneck,resnets[x],channels) 
    CNNresnet = nn.Sequential(
        basicResnet,
        nn.Sigmoid()
    )
    return CNNresnet

# model= GrayscaleResNet(torchvision.models.resnet.Bottleneck,[3, 4, 6, 3])
# [3, 4, 6, 3] => https://github.com/pytorch/vision/blob/791c172a337d98012018f98ffde93b1020ba3ed5/torchvision/models/resnet.py#L236