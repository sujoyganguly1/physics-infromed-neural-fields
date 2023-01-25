# Base neural network for PINNs

import torch
from collections import OrderedDict
import abc
import numpy as np


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class PhysicsInformedNN():
    __metaclass__ = abc.ABCMeta
    def __init__(self, X, u, layers, device='cpu'):
        self.device = device
        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u).float().to(self.device)

        self.dnn = DNN(layers).to(self.device)

        @abc.abstractmethod
        def net_u(self, x, t):
            pass
        @abc.abstractmethod
        def net_f(self, x, t):
            pass
        @abc.abstractmethod
        def losses(self):
            pass
        @abc.abstractmethod
        def train(self, num_iter):
            pass
        @abc.abstractmethod
        def predict(self, X):
            pass
