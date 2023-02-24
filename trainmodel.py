import projetLib as proj
from torch.utils.data import DataLoader
import torch
import sys
from torch.nn import BCEWithLogitsLoss, HingeEmbeddingLoss
from statistics import mean

runName = "smallpe_2"
batch_size = 16
limit = 10000

epochs = 1
if len(sys.argv)>1:
    epochs = int(sys.argv[1])

trainDataset, testDataset = proj.image.getTrainTest(
    resize=(224,224), batch_size=batch_size, seed=1, limit=limit,
    test_proportion=0.2, extensions=["pe"])

print(f"{runName} : Images de train {len(trainDataset)}, Images de test {len(testDataset)}")

trainloader = DataLoader(trainDataset, num_workers=2, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testDataset, num_workers=2, batch_size=batch_size, shuffle=True)

model =  proj.model.getCNNresnet()
# model = proj.model.VGG16()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)

# tuples de loss et leur coef
losses = [
    #(imp.loss.perceptualVGG,1),
    #(imp.loss.totalVariation,1),
    (BCEWithLogitsLoss(),1)
    # (HingeEmbeddingLoss(),1)
]

proj.process.train_malware(model, optimizer, trainloader, losses, testloader, runName=runName, epochs=epochs)