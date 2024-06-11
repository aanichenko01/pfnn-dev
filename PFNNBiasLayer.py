
import torch
import torch.nn as nn

class PFNNBiasLayer(nn.Module):
    def __init__(self, shape):
        super(PFNNBiasLayer, self).__init__()
        self.b = nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.b