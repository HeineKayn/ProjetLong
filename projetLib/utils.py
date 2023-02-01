from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from statistics import mean

from torchvision import transforms
import torch
import os

__all__ = ["plot_img"]

def plot_img(x,title=""):
    """ Affiche les images fournies en entrée
    - **x** : torch.Size([batch_size, **3**, w, h]) 
    - **title** : Si title n'est pas null l'affiche au dessus de l'image
    - **return** : `None`
    """

    img_grid = make_grid(x[:16])
    plt.figure(figsize=(20,15))
    plt.imshow(img_grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

class Visu :

    def __init__(self, expeName = "default", runName = "default", save = False, gridSize = 16):
        self.runName  = runName
        self.expeName = expeName
        self.path = "./savedImages/{}/{}.png".format(expeName,runName)
        self.save = save
        
        self.gridSize = gridSize
        self.count    = 0
        self.figSize  = (80,60)
        self.figSize  = (40,40)
        
    #--------- PLOT and save

    def plot_img(self,images,**kwargs):
        self.count += 1
        images = images[:,:3]
        images = torch.clip(images[:self.gridSize],0,1)
        img_grid = make_grid(images)
        plt.figure(figsize=self.figSize)
        plt.imshow(img_grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        
        if self.save :
            dir_path = "/".join(self.path.split("/")[:-1])
            if not os.path.exists(dir_path) :
                os.makedirs(dir_path)
                
            variable_path = self.path[:-4] + str(self.count) + self.path[-4:] 
            plt.savefig(variable_path)
            

    def plot_original_img(self,**kwargs):
        self.plot_img(kwargs["x"])
        
    def plot_altered_img(self,**kwargs):
        self.plot_img(kwargs["x_prime"])
        
    def plot_res_img(self,**kwargs):
        self.plot_img(kwargs["x_hat"])
        
    def plot_all_img(self,**kwargs):
        self.plot_original_img(kwargs)
        self.plot_altered_img(kwargs)
        self.plot_res_img(kwargs)
        
   #--------- PLOT and save     
        
    def indiv_img(self,images,title,**kwargs):
        for i,image in enumerate(images) : 
            image = transforms.ToPILImage()(image)
            path = "./savedImages/yanis/{}{}.jpg"
            image.save(path.format(title,str(i)))
            
            
    def indiv_original_img(self,**kwargs):
        self.indiv_img(kwargs["x"],"original")
        
    def indiv_altered_img(self,**kwargs):
        self.indiv_img(kwargs["x_prime"],"altered")
        
    def indiv_res_img(self,**kwargs):
        self.indiv_img(kwargs["x_hat"],"res")
        
    #--------- BOARD
        
    def board_plot_img(self,**kwargs):
        images_prime = kwargs["x_prime"].cuda()
        images_hat   = kwargs["x_hat"].cuda()
        
        dir_path = "runs/{}/altered".format(self.expeName)
#         if not os.path.exists(dir_path) :
        writer  = SummaryWriter(dir_path)
        images = images_prime[:self.gridSize]
        images = torch.clip(images,0,1)
        img_grid = make_grid(images)
        writer.add_image("Altered",img_grid)
        writer.close()
        
        writer  = SummaryWriter("runs/{}/{}".format(self.expeName,self.runName))
        images = images_hat[:self.gridSize]
        images = torch.clip(images,0,1)
        img_grid = make_grid(images)
        writer.add_image("Output",img_grid)
        
        writer.close
        
    def board_loss_train(self,**kwargs):
        running_loss = kwargs["running_loss"]
        epoch        = kwargs["epoch"]
        
        writer = SummaryWriter("runs/{}/{}".format(self.expeName,self.runName))
        writer.add_scalar("training loss", mean(running_loss), epoch)
        writer.close()
        
    def board_loss_test(self,**kwargs):
        running_loss = kwargs["running_loss"]
        
        writer = SummaryWriter("runs/{}/{}".format(self.expeName,self.runName))
        writer.add_text("testing loss", str(mean(running_loss)))
        writer.close()
        
        
    def none(self,**kwargs):
        pass