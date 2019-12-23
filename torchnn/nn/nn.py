import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, inp, hidd, out):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            *self._make_module(inp,hidd),
            *self._make_module(hidd,out),
            nn.Tanh()
        )
    
    def _make_module(self, inNum, outNum):
        return [nn.Linear(inNum,outNum), nn.LeakyReLU(0.2)]
    
    def forward(self, x):
        return self.model(x)

x = torch.randn(10000, 100)
y = torch.randn(10000, 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN(100,50,20).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(1000):
    pred = model(x)

    loss_fn = loss(pred, y)
    print(i, loss_fn.item())
    optimizer.zero_grad()
    loss_fn.backward()
    optimizer.step()

