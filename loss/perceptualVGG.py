import torch 
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["perceptualVGG"]

def gram_matrix(input):
    a, b, c, d = input.size()   # a=batch size(=1)
                                # b=number of feature maps
                                # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d) 
    G = torch.mm(features, features.t())  # compute the gram product
    
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        vgg_model = models.vgg16(pretrained=True).to(device).eval()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        features = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                features.append(x)
        return features
    
loss_network = LossNetwork()
    
def perceptualVGG(x, x_hat):
    """ Compare l'image originale et la reconstruite et essaie de les faire tendre vers un visage normale
    - **x** : torch.Size([batch_size, c, w, h])
    - **x_hat** : torch.Size([batch_size, c, w, h])
    - **return** : int"""

    mse = torch.nn.MSELoss()
    x_feats = loss_network(x)
    with torch.no_grad():
        x_hat_feats = loss_network(x_hat)
    #content loss:
    loss = mse(x_feats[2], x_hat_feats[2].detach())
    #style loss:
    for feats, target_feats in zip(x_feats, x_hat_feats):
        loss += mse(gram_matrix(feats), gram_matrix(target_feats.detach()))
    return loss