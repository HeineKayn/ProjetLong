import projetLib as proj
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset
import torchvision
from tqdm import tqdm
import sys
from torch.nn import BCEWithLogitsLoss
from statistics import mean

runName = "first"
batch_size = 32

trainDataset, testDataset = proj.data.getTrainTest(
    resize=(224,224), batch_size=batch_size, seed=1,
    test_proportion=0.2, extensions=["pe","msdos","elf","other"])

trainloader = DataLoader(trainDataset, num_workers=2, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testDataset, num_workers=2, batch_size=batch_size, shuffle=True)

CNNresnet = proj.model.GrayscaleResNet(torchvision.models.resnet.Bottleneck,[3, 4, 6, 3])
optimizer = torch.optim.Adam(CNNresnet.parameters(), lr=1e-3, weight_decay=0.001)

# tuples de loss et leur coef
losses = [#(imp.loss.perceptualVGG,1),
          #(imp.loss.totalVariation,1),
          (BCEWithLogitsLoss(),1)]

proj.process.train_malware(CNNresnet, optimizer, trainloader, losses, testloader, runName=runName, epochs=5)
final_acc = proj.process.test_malware(CNNresnet, testloader)
print("Accuracy Finale : {}".format(final_acc))