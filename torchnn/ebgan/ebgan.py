import argparse
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

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
         self.embedding = nn.Linear(down_dim, 32)
    
    def forward(self, img, labels):
       pass

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
generator = Generator(opt.classes, opt.latent_dim, (opt.channels, opt.img_size, opt.img_size))
discriminator = Discriminator(opt.classes, opt.latent_dim, (opt.channels, opt.img_size, opt.img_size))

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
    for i, (imgs, labels) in enumerate(trainloader):
        batch_size = imgs.shape[0]

        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        img = Variable(imgs.type(torch.FloatTensor))
        labels = Variable(labels.type(torch.LongTensor))
        optimizer_G.zero_grad()

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(torch.LongTensor(np.random.randint(0, opt.classes, batch_size)))
        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        optimizer_D.zero_grad()
        disk_images = discriminator(img, labels)
        a_loss_real = adversarial_loss(validity, valid)

        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        a_loss_fake = adversarial_loss(validity_fake, fake)

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