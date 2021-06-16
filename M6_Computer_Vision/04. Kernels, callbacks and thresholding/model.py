import matplotlib.pyplot as plt
import time
import numpy as np

import torch.nn as nn
import torch
from torch.nn import Conv2d,  MaxPool2d, ReLU, Sequential, BatchNorm2d, Dropout, Module, Linear
from torch import optim
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Grayscale
import torch.nn.functional as F

class Net(Module):
    def __init__(self):
        super(Net,self).__init__()

        self.cnn_layers = Sequential(

            Conv2d(1,128,kernel_size = 3, stride = 1, padding = 1),
            Dropout(0.4, inplace=True),
            BatchNorm2d(128),
            ReLU(inplace=True),

            Conv2d(128,64,kernel_size = 3, stride = 1),
            Dropout(0.4, inplace=True),
            BatchNorm2d(64),
            ReLU(inplace=True),

            Conv2d(64,32,kernel_size= 3, stride = 2),
            Dropout(0.4, inplace=True),
            BatchNorm2d(32),
            ReLU(),
        )

        self.linear_layers = Sequential(

            Linear(32 * 12 * 12, 64),
            Dropout(0.4, inplace=True),
            ReLU(),
            Linear(64, 32),
            Dropout(0.4, inplace=True),
            ReLU(),
            Linear(32, 16),
            Dropout(0.4, inplace=True),
            ReLU(),
            Linear(16, 10),
        )
    
    def forward(self, x):

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x