import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

class Autoencoder(nn.Module):
    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            *self._block(params),
        )
        self.decoder =  nn.Sequential(
            *self._block(list(reversed(list(map(lambda x: (x[1], x[0]), params))))),
            nn.Tanh()
        )
    
    def _block(self, layers_inp):
        layers = []
        for layer in layers_inp:
            layer_param = nn.Linear(layer[0], layer[1])
            nn.init.xavier_uniform_(layer_param.weight)
            layers.append(layer_param)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)