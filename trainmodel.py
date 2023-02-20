import projetLib as proj
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset
import torchvision
from tqdm import tqdm
import sys
from torch.nn import BCEWithLogitsLoss
from statistics import mean

size = (224,224)
batch_size = 32
test_proportion = 0.2
seed = 1
runName = "first"

dataset = proj.data.allImageDataset(size) # ,["msdos"]
lenTrainTest = int(len(dataset)*(1-test_proportion))
restDataset  = lenTrainTest%batch_size
g = torch.Generator()
if seed != 0 :
    g.manual_seed(seed)

trainDataset,testDataset = torch.utils.data.random_split(dataset, [lenTrainTest-restDataset, len(dataset)-lenTrainTest+restDataset],g)
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