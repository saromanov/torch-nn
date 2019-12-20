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
    def __init__(self, latent_dim, shape):
        super(Generator, self).__init__()
        self._shape = shape
        self.model = nn.Sequential(
            *self._block(latent_dim, 128),
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
    
    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *self._shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, dims, shape):
        super(Discriminator, self).__init__()
        self._shape = shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1),
        )
    
    def forward(self, img):
        data = img.view(img.shape[0], -1)
        return self.model(data)

def boundary_loss(valid, pred):
    return 0.5 * torch.mean((torch.log(pred) - torch.log(pred))**2)

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

adversarial_loss = torch.nn.MSELoss()
generator = Generator(opt.latent_dim, (opt.channels, opt.img_size, opt.img_size))
discriminator = Discriminator(opt.latent_dim, (opt.channels, opt.img_size, opt.img_size))

trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.d1, opt.d2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.d1, opt.d2))

for epoch in range(opt.epochs):
    for i, (imgs, _) in enumerate(trainloader):
        image = Variable(imgs.type(torch.FloatTensor))

        valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        img = Variable(imgs.type(torch.FloatTensor))
        optimizer_G.zero_grad()

        # Sample noice
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = generator(z)
        bs_loss = boundary_loss(discriminator(gen_imgs), valid)
        bs_loss.backward()
        optimizer_G.step()
        optimizer_D.zero_grad()

        validity_real = discriminator(image)
        a_loss_real = adversarial_loss(validity_real, valid)

        validity_fake = discriminator(gen_imgs.detach())
        a_loss_fake = adversarial_loss(validity_fake, fake)

        d_loss = 0.5 * (a_loss_real + a_loss_fake)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.epochs, i, len(trainloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(trainloader) + i
        #if batches_done % opt.sample_interval == 0:
        #    sample_image(n_row=10, batches_done=batches_done)

