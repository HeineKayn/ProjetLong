import torch.nn as nn
import torch 

__all__ = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Basic(nn.Module):
    
    def __init__(self,in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            # nn.ReLU(True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2),
            # nn.ReLU(True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=1),
            # nn.ReLU(True),
        )
        self.linear = nn.Linear(27*27*128, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = nn.Sigmoid()(x)
        return x