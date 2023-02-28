import projetLib as proj
from torch.utils.data import DataLoader
import torch
import sys
from torch.nn import BCEWithLogitsLoss, HingeEmbeddingLoss
from statistics import mean

runName = "224_resize"
batch_size = 16
resize = (224,224)
sizeTrain = [5000,5000]
sizeTest = [1000,1000]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 5
if len(sys.argv)>1:
    epochs = int(sys.argv[1])

trainDataset, testDataset = proj.image.getTrainTest(
    resize=resize, batch_size=batch_size, trainSize=sizeTrain, testSize=sizeTest,
    extensions=["pe"], seed=1, doRGB=False, doCrop=False)

print(f"{runName} : Images de train {len(trainDataset)}({int(sizeTrain[0]/len(trainDataset)*100)}% malware), Images de test {len(testDataset)} ({int(sizeTest[0]/len(testDataset)*100)}% malware)")

trainloader = DataLoader(trainDataset, num_workers=2, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testDataset, num_workers=2, batch_size=batch_size, shuffle=True)

model =  proj.model.getCNNresnet(101,channels=1)
# model =  proj.model.Basic()
# model = proj.model.VGG16(input_channel=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)

# tuples de loss et leur coef
losses = [
    #(imp.loss.perceptualVGG,1),
    #(imp.loss.totalVariation,1),
    (BCEWithLogitsLoss(),1)
    # (HingeEmbeddingLoss(),1)
]

proj.process.train_malware(model, optimizer, trainloader, losses, testloader, runName=runName, epochs=epochs)