import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# custom loss function from PFNN paper
def loss_func(output, target, model):
    loss = torch.mean((output - target)**2) + model.cost()
    return loss

# training loop for PFNN
def train_pfnn(model, data_loader, optimizer, num_epochs=10, device='cuda'):
    for epoch in range(num_epochs):
        model.train()
        loss_list = []
        for i, batch in enumerate(tqdm(data_loader)):
            input, target = batch
            input, target = input.to(device), target.to(device)

            # forward pass
            output = model(input)
            loss = loss_func(output, target, model)
            loss_list.append(loss.item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.average(loss_list)}')
    
    return model