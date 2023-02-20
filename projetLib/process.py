import torch
from tqdm import tqdm
import sys
from statistics import mean
from torch import nn
from torchmetrics import ConfusionMatrix

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
        t2 = tqdm(loader, leave=False, colour="#005500", file=sys.stdout) 
        net.train()
        for batch_idx,(x,y) in enumerate(t2):
            x = x.to(device)
            y = y.to(device).float()
            with torch.set_grad_enabled(True):
                outputs = net(x)
                try :
                    outputs  = torch.reshape(outputs,(32,))
                    loss = 1e-5
                    for criterion,coef in losses : 
                        loss += criterion(outputs, y)*coef
                    loss /= accum_iter
                    running_loss.append(loss.item())
                    loss.backward()
                    if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(t2)):
                        optimizer.step()
                        optimizer.zero_grad()
                    t2.set_description(f'Training loss: {mean(running_loss)*1000:.5f}, LR : {current_lr}, epoch {epoch + 1}/{epochs}')
                except Exception as e:
                    pass

        accuracy = test_malware(net, testloader)
        t1.set_description(f'Epoch {epoch + 1}/{epochs}, Accuracy {accuracy}, LR : {current_lr}')
        if lrDecrease :        
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            scheduler.step()
        torch.save(net.state_dict(),"./modelSave/{}_{}".format(runName,epoch))


def test_malware(net, testloader):
    with torch.no_grad():
        net.eval()
        matrix = torch.zeros((2,2))
        confmat = ConfusionMatrix(task="binary", num_classes=2)
        for x, y in testloader : 
            outputs = net(x)
            outputs = torch.reshape(outputs,(len(y),))
            m = nn.Sigmoid()
            outputs = m(outputs)
            matrix  += confmat(outputs, y)
        (tp,fp),(fn,tn) = matrix
        accuracy = (tp+tn)/(tp+tn+fp+fn) 
        return accuracy

        #recall = tp/(tp+fn) # quelle proporition positif a été id correctement 
        #precision = tp/(tp+fp) # quelle proportion positif = correct