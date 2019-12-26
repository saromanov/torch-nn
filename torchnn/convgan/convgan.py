
import argparse
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Generator(nn.Module):
    ''' Definition of the generator class
    '''
    def __init__(self, size, feature):
        super(Generator, self).__init__()
        self._shape = shape
        self.fc = nn.Linear(input_size, num_feature)
        self.model = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes, dims, shape):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.conv1 = self._make_conv_layer(1,32,5)
        self.conv2 = self._make_conv_layer(32,64,5)
    
    def _make_conv_layer(x,y,z):
        return nn.Sequential(
            nn.Conv2d(x, y, z, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc(x.view(x.size(0), -1))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

D = discriminator().cuda()
G = generator(z_dimension, 3136).cuda()

