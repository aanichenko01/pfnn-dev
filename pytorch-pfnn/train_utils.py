import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

# custom loss function from PFNN paper
def loss_func(output, target, model):
    loss = torch.mean((output - target)**2) + model.cost()
    return loss

# training loop for PFNN with early stopping
def train_pfnn_patience(model, data_loader, optimizer, num_epochs=10, device='cuda', patience=3):
    best_loss = float('inf')
    best_model = None
    loss_hist = []
    patience_count = 0

    for epoch in range(num_epochs):
        model.train()
        # reset loss list every epoch
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
        
        # track and print loss stats
        avg_loss = np.average(loss_list)
        loss_hist.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.average(loss_list)}')

        # trakc best model + early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model)
            # reset patience
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            print(f"Training stopped after {patience} epochs with no improvement.")
            break

    return best_model, loss_hist

def train_pfnn_thresh(model, data_loader, optimizer, num_epochs=10, device='cuda', threshold=0.01):
    best_loss = float('inf')
    best_model = None
    loss_hist = []

    for epoch in range(num_epochs):
        model.train()
        # reset loss list every epoch
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
        
        # track and print loss stats
        avg_loss = np.average(loss_list)
        loss_hist.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.average(loss_list)}')

        loss_diff = abs(avg_loss - best_loss)
        print(loss_diff)
        # track best model + early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model)

        if loss_diff < threshold:
            print(f"Training stopped as loss did not improve by set threshold of {threshold}.")
            break

    return best_model, loss_hist