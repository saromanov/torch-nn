import argparse
import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

if not os.path.exists('./ebgan_img'):
    os.mkdir('./ebgan_img')

class Discriminator(nn.Module):
    '''
    Discriminator structed as autoencoder
    In the case of images using upsampling and downsampling
    '''
    def __init__(self, channels, size):
        super(Discriminator, self).__init__()
        self._downsampling = nn.Sequential(
            nn.Conv2d(opt.channels, 64, 3, 2, 1),
            nn.ReLU()
        )
        dim = self._create_down(size)
        self.embedding = nn.Linear(dim, 32)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
    
    def _create_down(self, size):
        return 64 * (opt.img_size // 2) ** 2 
    
    def forward(self, img, labels):
       out = self.down(img)
       embedding = self.embedding(out.view(out.size(0), -1))
       out = self.fc(embedding)
       out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
       return out, embedding

class Generator(nn.Module):
    def __init__(self, channels, dims, size):
        super(Generator, self).__init__()
        self._size = size // 4
        self.l1 = nn.Sequential(nn.Linear(dims, 128 * (opt.img_size // 4) ** 2))
        self.l2 = nn.Sequential(
            *self._block(128,128,3),
            *self._block(128,64,3),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def _block(self, w,h,d):
        return [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(w, h, d, stride=1, padding=1),
            nn.BatchNorm2d(w, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ]
    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self._size, self._size)
        return self.l2(out)

def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt

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
adversarial_loss = torch.nn.MSELoss()
generator = Generator(opt.channels, opt.latent_dim, opt.img_size)
discriminator = Discriminator(opt.channels, opt.img_size)

trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.d1, opt.d2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.d1, opt.d2))

lambda_pt = 0.1
for epoch in range(opt.epochs):
    for i, (imgs, _) in enumerate(trainloader):
        batch_size = imgs.shape[0]

        real = Variable(imgs.type(torch.FloatTensor))
        optimizer_G.zero_grad()

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_imgs = generator(z)
        validity = discriminator(gen_imgs)

        g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)

        optimizer_D.zero_grad()
        g_loss.backward()

        disk_images = discriminator(real)
        validity_fake = discriminator(gen_imgs.detach())
        a_loss_real = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)
        a_loss_fake = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)

        d_loss = (a_loss_real + a_loss_fake) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.epochs, i, len(trainloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(trainloader) + i
        #if batches_done % opt.sample_interval == 0:
        #    sample_image(n_row=10, batches_done=batches_done)