import torch
from statistics import mean
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision import transforms
from torchvision.utils import make_grid

import numpy as np
import impaintingLib as imp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_inpainting(net, optimizer, loader, alter, losses, runName="bigRun", scale_factor=4, epochs=5, lrDecrease=True, simplify_seg=True, show_images=True, summary=True):
    """Voir le tutoriel pour une explication en détail"""
    net.train()
    accum_iter = 100 
    lrs = []
    current_lr = optimizer.param_groups[0]["lr"]
    lambda1 = lambda epoch: 0.67 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    t1 = tqdm(range(epochs), desc=f"Training progress", colour="#00ff00")
    
    for epoch in t1:
        running_loss = []
        t2 = tqdm(loader, leave=False, colour="#005500") 

        for batch_idx,(x,_) in enumerate(t2):
            x = x.to(device)
            #x = imp.data.randomTransfo(x)
            x_prime = alter(x)
            
            with torch.set_grad_enabled(True):
                segmented = imp.components.get_segmentation(x, simplify=simplify_seg, scale_factor=scale_factor)
                x_input = torch.cat((x_prime, segmented),dim=1)
                keypointLayer = imp.components.getKeypoints(x)
                x_input = torch.cat((x_input, keypointLayer),dim=1)

                outputs = net(x_input)

                loss = 1e-5
                for criterion,coef in losses : 
                    loss += criterion(outputs, x)*coef
                loss /= accum_iter

                running_loss.append(loss.item())
                loss.backward()

                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(t2)):
                    optimizer.step()
                    optimizer.zero_grad()
                
                t2.set_description(f'Epoch {epoch}, training loss: {mean(running_loss)}, LR : {current_lr}, epoch {epoch + 1}/{epochs}')

        t1.set_description(f'Epoch {epoch + 1}/{epochs}, LR : {current_lr}')

        if lrDecrease :        
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            scheduler.step()
            
        if show_images:
            imp.utils.plot_img(x[:8])
            imp.utils.plot_img(x_prime[:8])
            imp.utils.plot_img(segmented[:8])
            imp.utils.plot_img(torch.clip(outputs[:8], 0, 1))
            imp.utils.plot_img(keypointLayer[:8])
            
        if summary:
            writer = SummaryWriter("runs/" + runName)
            writer.add_scalar("training loss", mean(running_loss), epoch)
            writer.add_image("Original",make_grid(x[:8]))
            writer.add_image("Mask",make_grid(x_prime[:8]))
            writer.add_image("Predict",make_grid(torch.clip(outputs[:8], 0, 1)))
            writer.close()
        
        torch.save(net.state_dict(),"./modelSave/train/{}_{}".format(runName,epoch))