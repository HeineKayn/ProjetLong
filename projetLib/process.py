import torch
from tqdm import tqdm
import sys
from statistics import mean
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_malware(net, optimizer, loader, losses, runName="default", epochs=5, lrDecrease=True):
    net = net.to(device) 
    net.train()
    accum_iter = 100 
    lrs = []
    current_lr = optimizer.param_groups[0]["lr"]
    lambda1 = lambda epoch: 0.67 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    t1 = tqdm(range(epochs), total=epochs, desc=f"Training progress", colour="#00ff00", leave=True, file=sys.stdout)    
    for epoch in t1:
        running_loss = []
        t2 = tqdm(loader, leave=False, colour="#005500", file=sys.stdout) 
        for batch_idx,(x,y) in enumerate(t2):
            x = x.to(device)
            y = y.to(device).float()
            with torch.set_grad_enabled(True):
                outputs = net(x)

                try :
                    outputs = torch.reshape(outputs,(32,))
                    outputs = (outputs-torch.mean(outputs))/torch.std(outputs)
                    # predicted = torch.argmax(outputs, dim=1)
                    # print(predicted)
                    loss = 1e-5
                    for criterion,coef in losses : 
                        loss += criterion(outputs, y)*coef
                    loss /= accum_iter
                    running_loss.append(loss.item())
                    loss.backward()
                    if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(t2)):
                        optimizer.step()
                        optimizer.zero_grad()
                    t2.set_description(f'Epoch {epoch}, training loss: {mean(running_loss)}, LR : {current_lr}, epoch {epoch + 1}/{epochs}')
                except Exception as e:
                    pass
                    # print(e)
                    # print(outputs.shape)
        t1.set_description(f'Epoch {epoch + 1}/{epochs}, LR : {current_lr}')

        if lrDecrease :        
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            scheduler.step()
        torch.save(net.state_dict(),"./modelSave/{}_{}".format(runName,epoch))


def test_malware(net, testloader):
    with torch.no_grad():
        positive = 0.0
        for x,y in testloader : 
            outputs = net(x)
            m = nn.Sigmoid()
            outputs = m(outputs)
            positive += torch.sum(outputs == y)
        positive /= len(testloader.dataset)
        print(positive)