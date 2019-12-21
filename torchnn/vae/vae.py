import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.layer1 = self._block(input_dim, 300)
        self.layer2 = self._block(300, 20)
        self.layer3 = self._block(300, 20)
        self.layer4 = self._block(20, 300)
        self.layer5 = self._block(300, input_dim)
    
    def _block(self, in_features, out_features, batch_norm=0.8):
        return nn.Linear(in_features, out_features)
    
    def _encode(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden), self.layer3(hidden)
    
    def _decode(self, x):
        return torch.sigmoid(self.layer5(F.relu(self.layer4(x))))
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self._encode(x.view(-1, 784))
        z = self._reparameterize(mu, logvar)
        return self._decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=opt.batch_size, shuffle=True)

model = VAE(784).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

for epoch in range(opt.epochs):
    model.train()
    train_loss = 0
    for i, (imgs, _) in enumerate(trainloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(imgs)
        loss = loss_function(recon_batch, imgs, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(trainloader.dataset)))
