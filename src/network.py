# Base neural network for PINNs

import torch
from collections import OrderedDict
import abc
import numpy as np


class DNN(torch.nn.Module):
    def __init__(self, layers):
        # CUDA support
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
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


class PhysicsInformedNN(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, u, layers, lb, ub):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u).float().to(self.device)

        # settings
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(self.device)
        self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(self.device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # deep neural networks
        self.dnn = DNN(layers).to(self.device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

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
