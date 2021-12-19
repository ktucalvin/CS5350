import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetworkClassifier(nn.Module):
    def __init__(self, depth, width, dims, activation=nn.Sigmoid):
        super(NeuralNetworkClassifier, self).__init__()

        stack = [nn.Linear(dims, width), activation(), nn.BatchNorm1d(width)]
        for _ in range(depth - 1):
            stack.append(nn.Linear(width, width))
            stack.append(activation())
            stack.append(nn.BatchNorm1d(width))
        stack.append(nn.Linear(width, 1))
        stack.append(nn.Sigmoid())
        self.network = nn.Sequential(*stack)
    
    def init_weights(self, strategy="xavier"):
        if strategy == "xavier":
            self.network.apply(self.init_xavier)
        elif strategy == "he":
            self.network.apply(self.init_he)
    
    def init_xavier(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def init_he(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)
    