# -*- coding: utf-8 -*-
"""
@author: Clément Bonnet
"""

import torch
import argparse
import matplotlib
from torchvision.utils import save_image
import torchvision.transforms as transforms
import model
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
from torchvision import datasets
import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
matplotlib.style.use('ggplot')
from torchvision.utils import make_grid

EPOCHS = 80
BATCH_SIZE = 64
Lr = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='../input/data',train=True,download=True,transform=transform)
val_data = datasets.MNIST(root='../input/data',train=False,download=True,transform=transform)
train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_data,batch_size=BATCH_SIZE,shuffle=False)
model = model.LinearVAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=Lr)
criterion = nn.BCELoss(reduction='sum')

def Loss(loss, mu, logvar):

    KullBackLeiblerLoss = -torch.sum(-mu.pow(2)-logvar.exp()+1+logvar )/2
    return loss + KullBackLeiblerLoss

def fit(model, dataloader):
    model.train()
    current_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(DEVICE)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        res, mu, logVariance = model(data)
        loss = Loss(criterion(res, data), mu, logVariance)
        current_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = current_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader):
    model.eval()
    current_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, _ = data
            data = data.to(DEVICE)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            loss = Loss(criterion(reconstruction, data), mu, logvar)
            current_loss += loss.item()

            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = reconstruction.view(BATCH_SIZE, 1, 28, 28)[:64]
                print(both.shape)
                name = str(epoch)
                save_image(both.cpu(), "/results/"+name+".png", nrow=num_rows)
    val_loss = current_loss/len(dataloader.dataset)
    return val_loss, both

grid_images = []

for epoch in range(EPOCHS):
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss, recon_images = validate(model, val_loader)
    image = make_grid(recon_images.detach().cpu())
    grid_images.append(image)