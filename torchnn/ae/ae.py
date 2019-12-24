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
            *self._block(list(reversed(map(lambda x: (x[1], x[0]), params))), normalize=False),
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

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--input_dim", type=int, default=100, help="dimensionality of the input space")
parser.add_argument("--hidden_dim", type=int, default=50, help="dimensionality of the hidden space")
parser.add_argument("--out_dim", type=int, default=20, help="dimensionality of the output space")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(opt.epochs):
    optimizer.zero_grad()
    loss_fn.backward()
    optimizer.step()