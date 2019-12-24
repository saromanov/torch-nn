import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

class Autoencoder(nn.Module):
    def __init__(self, params):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            *self._block(params, normalize=False),
        )
        self.decoder =  nn.Sequential(
            *self._block(dims + num_classes, 128, normalize=False),
            *self._block(128, 256),
            *self._block(256, 512),
            *self._block(512, 1024),
            nn.Linear(1024, int(np.prod(self._shape))),
            nn.Tanh()
        )
    
     def _block(self, layers_inp):
         layers = []
         for layer in layers_inp:
             layers.append(nn.Linear(layer[0], layer[1]))
             layers.append(nn.LeakyReLU(0.2, inplace=True))
         return layers