# -*- coding: utf-8 -*-
"""
@author: Clément Bonnet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

KERNEL_SIZE = 4 
CHANNELS_AT_BEGINNING= 8
LATENT_DIM = 20 

class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
        self.E1 = nn.Linear(in_features=784, out_features=512)
        self.E2 = nn.Linear(in_features=512, out_features=LATENT_DIM*2)
        self.D1 = nn.Linear(in_features=LATENT_DIM, out_features=512)
        self.D2 = nn.Linear(in_features=512, out_features=784)
 
    def forward(self, y):

        y = self.E2(F.relu(self.E1(y))).view(-1, 2, LATENT_DIM)
        mu = y[:, 0, :]
        logVariance = y[:, 1, :]
        z = mu + (torch.randn_like(torch.exp(0.5*logVariance)) * torch.exp(0.5*logVariance))
        y = F.relu(self.D1(z))
        res = torch.sigmoid(self.D2(y))
        return res, mu, logVariance

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.E1 = nn.Conv2d(in_channels=1,out_channels=CHANNELS_AT_BEGINNING,kernel_size=KERNEL_SIZE, tride=2, padding=1)
        self.E2 = nn.Conv2d(in_channels=CHANNELS_AT_BEGINNING,out_channels=CHANNELS_AT_BEGINNING*2,kernel_size=KERNEL_SIZE,stride=2, padding=1)
        self.E3 = nn.Conv2d(in_channels=CHANNELS_AT_BEGINNING*2,out_channels=CHANNELS_AT_BEGINNING*4,kernel_size=KERNEL_SIZE,stride=2, padding=1)
        self.E4 = nn.Conv2d(in_channels=CHANNELS_AT_BEGINNING*4,out_channels=64,kernel_size=KERNEL_SIZE,stride=2, padding=0)
        self.appl1 = nn.Linear(64, 128)
        self.trouveParametreMu = nn.Linear(128, LATENT_DIM)
        self.trouveLogVariance = nn.Linear(128, LATENT_DIM)
        self.appl2 = nn.Linear(LATENT_DIM, 64)
        self.D1 = nn.ConvTranspose2d(in_channels=64,out_channels=CHANNELS_AT_BEGINNING*8,kernel_size=KERNEL_SIZE,stride=1, padding=0)
        self.D2 = nn.ConvTranspose2d(in_channels=CHANNELS_AT_BEGINNING*8,out_channels=CHANNELS_AT_BEGINNING*4,kernel_size=KERNEL_SIZE,stride=2, padding=1)
        self.D3 = nn.ConvTranspose2d(in_channels=CHANNELS_AT_BEGINNING*4,out_channels=CHANNELS_AT_BEGINNING*2,kernel_size=KERNEL_SIZE,stride=2, padding=1)
        self.D4 = nn.ConvTranspose2d(in_channels=CHANNELS_AT_BEGINNING*2,out_channels=1,kernel_size=KERNEL_SIZE,stride=2, padding=1)
 
    def Encoding(self, y):
        
        y = F.relu(self.E4(F.relu(self.E3(F.relu(self.E2(F.relu(self.E1(y))))))))
        batch, _, _, _ = y.shape
        y = F.adaptive_avg_pool2d(y, 1).reshape(batch, -1)
        hidden = self.appl1(y)
        mu = self.trouveParametreMu(hidden)
        logVariance = self.trouveLogVariance(hidden)
        return mu, logVariance
    
    def Decoding(self, z):

        return torch.sigmoid(self.D4(F.relu(self.D3(F.relu(self.D2(F.relu(self.dec1(z))))))))
    
    def forward(self, y):

        mu, logVariance = self.Encoding(y)
        z = self.appl2(mu + (torch.randn_like(std = torch.exp(0.5*logVariance)) * torch.exp(0.5*logVariance)))
        res = self.Decoding(z.view(-1, 64, 1, 1))
        
        return res, mu, logVariance