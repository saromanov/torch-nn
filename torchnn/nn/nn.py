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
            *self._make_module(hidd,out)
        )
    
    def _make_module(self, inNum, outNum):
        return [nn.Linear(inNum,outNum), nn.LeakyReLU(0.2)]
    
    def forward(self, x):
        return self.model(x)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--input_dim", type=int, default=100, help="dimensionality of the input space")
parser.add_argument("--hidden_dim", type=int, default=50, help="dimensionality of the hidden space")
parser.add_argument("--out_dim", type=int, default=20, help="dimensionality of the output space")
opt = parser.parse_args()

x = torch.randn(10000, opt.input_dim)
y = torch.randn(10000, opt.out_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN(opt.input_dim,opt.hidden_dim, opt.out_dim).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(opt.epochs):
    pred = model(x)

    loss_fn = loss(pred, y)
    print(i, loss_fn.item())
    optimizer.zero_grad()
    loss_fn.backward()
    optimizer.step()

