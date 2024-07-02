import torch
import torch.nn as nn
import numpy as np

from PFNNHiddenLayer import PFNNHiddenLayer
from Utils import cubic

class PhaseFunctionedNetwork(nn.Module):
    def __init__(self, input_shape=1, output_shape=1, dropout=0.7):
        super(PhaseFunctionedNetwork, self).__init__()

        self.nslices = 4
        self.dropout0 = nn.Dropout(p=dropout)
        self.hidden0 = PFNNHiddenLayer((self.nslices, 512, input_shape-1))
        self.activation0 = nn.ELU()
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.hidden1 = PFNNHiddenLayer((self.nslices, 512, 512))
        self.activation1 = nn.ELU()
        
        self.dropout2 = nn.Dropout(p=dropout)
        self.hidden2 = PFNNHiddenLayer((self.nslices, output_shape, 512))

        self.layers = nn.ModuleList([self.hidden0, self.hidden1, self.hidden2])

    def forward(self, input):
        pscale = self.nslices * input[:, -1]
        pamount = pscale % 1.0

        pindex_1 = torch.floor(pscale).long() % self.nslices
        pindex_0 = (pindex_1 - 1) % self.nslices
        pindex_2 = (pindex_1 + 1) % self.nslices
        pindex_3 = (pindex_1 + 2) % self.nslices

        Wamount = pamount.unsqueeze(1).unsqueeze(1)
        bamount = pamount.unsqueeze(1)
        
        # interpolate weights and biases for update using cubic Catmull-Rom spline function
        W0 = cubic(self.hidden0.W[pindex_0], self.hidden0.W[pindex_1], self.hidden0.W[pindex_2], self.hidden0.W[pindex_3], Wamount)
        b0 = cubic(self.hidden0.b[pindex_0], self.hidden0.b[pindex_1], self.hidden0.b[pindex_2], self.hidden0.b[pindex_3], bamount)

        W1 = cubic(self.hidden1.W[pindex_0], self.hidden1.W[pindex_1], self.hidden1.W[pindex_2], self.hidden1.W[pindex_3], Wamount)
        b1 = cubic(self.hidden1.b[pindex_0], self.hidden1.b[pindex_1], self.hidden1.b[pindex_2], self.hidden1.b[pindex_3], bamount)
        
        W2 = cubic(self.hidden2.W[pindex_0], self.hidden2.W[pindex_1], self.hidden2.W[pindex_2], self.hidden2.W[pindex_3], Wamount)
        b2 = cubic(self.hidden2.b[pindex_0], self.hidden2.b[pindex_1], self.hidden2.b[pindex_2], self.hidden2.b[pindex_3], bamount)

        # forward propogation
        x = input[:,:-1]
        x = self.dropout0(x.unsqueeze(-1))
        x = self.activation0(self.hidden0(W0, b0, x))

        x = self.dropout1(x.unsqueeze(-1))
        x = self.activation1(self.hidden1(W1, b1, x))

        x = self.dropout2(x.unsqueeze(-1))
        x = self.hidden2(W2, b2, x)

        return x
    
    def cost(self):
        costs = 0
        for layer in self.layers:
            if hasattr(layer, 'cost'):
                costs += layer.cost()
        return costs / len(self.layers)
    
    def precompute_and_save_weights(self):
        # precompute weights
        for i in range(50):
    
            pscale = self.nslices*(float(i)/50)
            pamount = pscale % 1.0
            
            pindex_1 = int(pscale) % self.nslices
            pindex_0 = (pindex_1-1) % self.nslices
            pindex_2 = (pindex_1+1) % self.nslices
            pindex_3 = (pindex_1+2) % self.nslices
            
            # interpolate weights and biases
            W0 = cubic(self.hidden0.W[pindex_0], self.hidden0.W[pindex_1], self.hidden0.W[pindex_2], self.hidden0.W[pindex_3], pamount).cpu().detach().numpy()
            b0 = cubic(self.hidden0.b[pindex_0], self.hidden0.b[pindex_1], self.hidden0.b[pindex_2], self.hidden0.b[pindex_3], pamount).cpu().detach().numpy()

            W1 = cubic(self.hidden1.W[pindex_0], self.hidden1.W[pindex_1], self.hidden1.W[pindex_2], self.hidden1.W[pindex_3], pamount).cpu().detach().numpy()
            b1 = cubic(self.hidden1.b[pindex_0], self.hidden1.b[pindex_1], self.hidden1.b[pindex_2], self.hidden1.b[pindex_3], pamount).cpu().detach().numpy()

            W2 = cubic(self.hidden2.W[pindex_0], self.hidden2.W[pindex_1], self.hidden2.W[pindex_2], self.hidden2.W[pindex_3], pamount).cpu().detach().numpy()
            b2 = cubic(self.hidden2.b[pindex_0], self.hidden2.b[pindex_1], self.hidden2.b[pindex_2], self.hidden2.b[pindex_3], pamount).cpu().detach().numpy()
            
            # save precomputed weights and biases
            W0.astype(np.float32).tofile('C:/Users/Ana/Desktop/dev/pfnn-dev/unity-pfnn/Assets/Demo/Dev/Weights/test/W0_%03i.bin' % i)
            b0.astype(np.float32).tofile('C:/Users/Ana/Desktop/dev/pfnn-dev/unity-pfnn/Assets/Demo/Dev/Weights/test/b0_%03i.bin' % i)

            W1.astype(np.float32).tofile('C:/Users/Ana/Desktop/dev/pfnn-dev/unity-pfnn/Assets/Demo/Dev/Weights/test/W1_%03i.bin' % i)
            b1.astype(np.float32).tofile('C:/Users/Ana/Desktop/dev/pfnn-dev/unity-pfnn/Assets/Demo/Dev/Weights/test/b1_%03i.bin' % i)
            
            W2.astype(np.float32).tofile('C:/Users/Ana/Desktop/dev/pfnn-dev/unity-pfnn/Assets/Demo/Dev/Weights/test/W2_%03i.bin' % i)
            b2.astype(np.float32).tofile('C:/Users/Ana/Desktop/dev/pfnn-dev/unity-pfnn/Assets/Demo/Dev/Weights/test/b2_%03i.bin' % i)