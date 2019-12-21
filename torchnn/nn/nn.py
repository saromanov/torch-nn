import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, inp, hidd, out):
        self.model = nn.Sequential(
            nn.Linear(inp,hidd),
            nn.ReLU(),
            nn.Linear(hidd, out)
        )
    
    def forward(self, x):
        return self.model(x)


loss = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

