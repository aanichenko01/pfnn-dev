import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PFNNHiddenLayer(nn.Module):
    def __init__(self, weights_shape, gamma=0.01):
        super(PFNNHiddenLayer, self).__init__()
        W_bound = np.sqrt(6. / np.prod(weights_shape[-2:]))
        self.W = nn.Parameter(torch.empty(weights_shape).uniform_(-W_bound, W_bound))
        self.b = nn.Parameter(torch.zeros((weights_shape[0], weights_shape[1])))
        self.gamma = gamma

    def forward(self, input):
        return torch.bmm(input, self.W.permute(0, 2, 1)) + self.b.unsqueeze(1)

    def cost(self):
        return self.gamma * torch.mean(torch.abs(self.W))