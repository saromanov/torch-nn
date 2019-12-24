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
            layers.append(nn.Linear(layer[0], layer[1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--input_dim", type=int, default=100, help="dimensionality of the input space")
parser.add_argument("--hidden_dim", type=int, default=50, help="dimensionality of the hidden space")
parser.add_argument("--out_dim", type=int, default=20, help="dimensionality of the output space")
parser.add_argument("--params", type=list, default=[(784,128), (128,64), (64,12), (12,3)], help="dimensionality of the output space")
opt = parser.parse_args()

trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
dataloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(opt.params).to(device)
loss = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

for i in range(opt.epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)

        output = model(img)
        loss_fn = loss(output, img)
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'
          .format(i + 1, opt.epochs, loss_fn))
    pic = to_img(output.cpu().data)
    save_image(pic, './mlp_img/image_{}.png'.format(i))