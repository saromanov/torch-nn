import argparse
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Generator(nn.Module):
    ''' Definition of the generator class
    '''
    def __init__(self, num_classes, dims, shape):
        super(Generator, self).__init__()
        self._shape = shape
        self.labels_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            *self._block(dims + num_classes, 128, normalize=False),
            *self._block(128, 256),
            *self._block(256, 512),
            *self._block(512, 1024),
            nn.Linear(1024, int(np.prod(self._shape))),
            nn.Tanh()
        )
    
    def _block(self, in_features, out_features, batch_norm=0.8):
        layers = [nn.Linear(in_features, out_features)]
        layers.append(nn.BatchNorm1d(out_features, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def forward(self, noise, labels):
        gen_input = torch.cat((self.labels_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self._shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, dims, shape):
        super(Discriminator, self).__init__()
        self._shape = shape
        self._labels_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1),
        )
    
    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self._labels_emb(labels)), -1)
        validity = self.model(d_in)
        return validity

def boundary_loss(valid, pred):
    return 0.5 * torch.mean((torch.log(pred) - K.log(pred))**2)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--d1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--d2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--cpus", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--channels", type=int, default=1, help="image channels")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
opt = parser.parse_args()

