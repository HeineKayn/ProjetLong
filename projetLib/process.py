import torch
from tqdm import tqdm
import sys
from statistics import mean
from torch import nn
from torchmetrics import ConfusionMatrix
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_malware(net, optimizer, loader, losses, testloader=[], runName="default", epochs=5, lrDecrease=True):
    net = net.to(device) 
    accum_iter = 100 
    lrs = []
    current_lr = optimizer.param_groups[0]["lr"]
    lambda1 = lambda epoch: 0.67 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    t1 = tqdm(range(epochs), total=epochs, desc=f"Training progress", colour="#00ff00", leave=True, file=sys.stdout)    
    for epoch in t1:
        running_loss = []
        t2 = tqdm(loader, leave=False, colour="#005500", leave=True, file=sys.stdout) 
        net.train()
        for batch_idx,(x,y) in enumerate(t2):
            x = x.to(device)
            y = y.to(device).float()
            with torch.set_grad_enabled(True):
                outputs = net(x)
                try :
                    outputs  = torch.reshape(outputs,(32,))
                    loss = 1e-5

                    criterion,coef = losses[0]
                    loss += criterion(outputs, y)*coef
                    outputs[outputs==0] = -1
                    criterion,coef = losses[1]
                    loss += criterion(outputs, y)*coef

                    # for criterion,coef in losses : 
                    #     loss += criterion(outputs, y)*coef
                    loss /= accum_iter
                    running_loss.append(loss.item())
                    loss.backward()
                    if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(t2)):
                        optimizer.step()
                        optimizer.zero_grad()
                    t2.set_description(f'Training loss: {mean(running_loss)*1000:.5f}')
                except Exception as e:
                    pass

        accuracy = test_malware(net, testloader)
        t1.set_description(f'Epoch {epoch + 1}/{epochs}, Accuracy {accuracy*100:.4f}%, LR : {current_lr}')
        if lrDecrease :        
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            scheduler.step()

        runFolder = f"./modelSave/{runName}/"
        if not os.path.exists(runFolder):
            os.makedirs(runFolder)
        torch.save(net.state_dict(),runFolder+f"model_{epoch}.pt")

        with open(runFolder + "last_results.txt","a") as f:
            f.write(f"{runName} - Epoch : {epoch} - Accuracy {accuracy*100:.4f}% - Loss : {mean(running_loss)*1000:.5f} LR : {current_lr}\n")

def test_malware(net, testloader):
    with torch.no_grad():
        net.eval()
        matrix = torch.zeros((2,2)).to(device)
        confmat = ConfusionMatrix(task="binary", num_classes=2).to(device)
        for x, y in testloader : 
            x = x.to(device)
            y = y.to(device)
            outputs = net(x)
            outputs = torch.reshape(outputs,(len(y),))
            matrix  += confmat(outputs, y)
        (tp,fp),(fn,tn) = matrix
        accuracy = (tp+tn)/(tp+tn+fp+fn) 
        return accuracy

        #recall = tp/(tp+fn) # quelle proporition positif a été id correctement 
        #precision = tp/(tp+fp) # quelle proportion positif = correct