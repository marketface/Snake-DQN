import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeDQN(nn.Module):
    def __init__(self):
        super(SnakeDQN, self).__init__()
        # all that's needed for this task is a simple little model with a two relu-activated hidden layers
        self.layer1 = nn.Linear(12, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 4)
        self.lerelu = nn.LeakyReLU()

        # hyperparams:
        # discount = .99
        # update freq = 1000
        # min eps = 0.01

        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=3e-4)
        print(sum(p.numel() for p in self.parameters()))
        
    def forward(self, x):
        # forward pass
        x = self.layer1(x)
        x = self.lerelu(x)
        x = self.layer2(x)
        x = self.lerelu(x)
        x = self.layer3(x)
        return x
    
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        '''Fit function to train the model on a single batch of data'''
        # moving data to GPU
        x,y = x.cuda(), y.cuda()
        # single-batch gradient descent update
        self.optim.zero_grad()
        predicted = self.__call__(x)
        loss = self.loss_fn(predicted, y)
        loss.backward()
        self.optim.step()
        return loss.item()

class SnakeConvDQN(nn.Module):
    def __init__(self):
        super(SnakeConvDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=6, stride=3)
        conv_w, conv_h = self.get_conv_shape(30, 30, 6, 3)
        # print(conv_w , conv_h , 64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        conv_w, conv_h = self.get_conv_shape(conv_w, conv_h, 3, 2)
        # print(conv_w , conv_h , 64)
        
        # print(conv_w * conv_h * 64)
        self.fc1 = nn.Linear(conv_w * conv_h * 64, 512) # input size is 1024
        self.fc2 = nn.Linear(512, 4)
        
        
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.parameters(), lr=2.5e-4)
        print(sum(p.numel() for p in self.parameters()))
        
    def forward(self, x: torch.Tensor):
        # layer 1
        x = self.conv1(x)
        x = F.leaky_relu(x)
        # layer 2
        x = self.conv2(x)
        x = F.leaky_relu(x)
        
        x = x.view(x.size(0), -1)
        
        # layer 3
        x = F.leaky_relu(self.fc1(x))
        # out
        x = self.fc2(x)
        
        return x
    
    def get_conv_shape(self, w, h, kernel, stride):
        next_w = (w - (kernel - 1) - 1) // stride + 1
        next_h = (h - (kernel - 1) - 1) // stride + 1
        return next_w, next_h
    
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        '''Fit function to train the model on a single batch of data'''
        print('fit')
        # moving data to GPU
        x,y = x.cuda(), y.cuda()
        # forward pass using nn.Module.__call__
        predicted = self.__call__(x)
        loss = self.loss_fn(predicted, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

# s = SnakeConvDQN()