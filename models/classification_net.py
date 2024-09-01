from torch import nn
import numpy as np
import torch

class Classification_net(nn.Module):
    def __init__(self, in_features,hidden_size):
        super(Classification_net, self).__init__()
        self.layer1=nn.Linear(in_features,hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.act=torch.nn.LeakyReLU()
        #self.layer1 = nn.Linear(in_features,1)

    def forward(self, x):
        x=self.layer1(x)
        x=self.act(x)
        x=self.layer2(x)


        print(x)
        return x

