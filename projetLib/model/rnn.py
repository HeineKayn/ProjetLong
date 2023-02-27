import torch.nn as nn
import torchvision

class RNN(nn.Module):
    
    def __init__(self,input_size,hidden_size,num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.dense = nn.Linear(hidden_size, 1)
        
    def forward(self, x): # x = [batch_size, 128, 49]
        # output, _ = self.lstm(x, (h0, c0))
        last_element = y[:,-1,:]
        y = self.dense(last_element)
        return y
    
    def __str__(self):
        return("RNN")