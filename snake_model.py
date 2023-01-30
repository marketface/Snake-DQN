import torch.nn as nn
from torch.utils.data import DataLoader
import torch

class SnakeDQN(nn.Module):
    def __init__(self):
        super(SnakeDQN, self).__init__()
        # all that's needed for this task is a simple little model with a two relu-activated hidden layers
        self.layer1 = nn.Linear(12, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
    
        # initializing optimizer inside the network since AdamW keeps internal state variables throughout training
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        
    def forward(self, x):
        # forward pass
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
    
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        '''Fit function to train the model on a single batch of data'''
        # initializing loss and optimizer
        loss_fn = nn.MSELoss()
        # moving data to GPU
        x,y = x.cuda(), y.cuda()
        # single-batch gradient descent update
        self.optim.zero_grad()
        predicted = self.forward(x)
        loss = loss_fn(predicted, y)
        loss.backward()
        self.optim.step()
        return loss.item()