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
    def __init__(self, num_classes, dims, shape):
        super(Generator, self).__init__()
        self._shape = shape
        self.labels_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            *self._block(dims + num_classes, 128, normalize=False),
            *self._block(128, 256),
            *self._block(256, 512),
            *self._block(512, 1024),
            nn.Linear(1024, int(np.prod(self._shape))),
            nn.Tanh()
        )
    
    def _block(self, in_features, out_features, normalize=True):
        layers = [nn.Linear(in_features, out_features)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_features, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def forward(self, noise, labels):
        gen_input = torch.cat((self.labels_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self._shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, dims, shape):
        super(Discriminator, self).__init__()
        self._shape = shape
        self._labels_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
        )
    
    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self._labels_emb(labels)), -1)
        validity = self.model(d_in)
        return validity


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

adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

for epoch in range(opt.epochs):
    for i, (img, _) in enumerate(trainloader):
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        
        num_img = img.size(0)
        real_img = Variable(img).to(device)
        real_label = Variable(torch.ones(num_img)).to(device)
        fake_label = Variable(torch.zeros(num_img)).to(device)

        real_out = D(real_img)
        d_loss_real = loss(real_out, real_label)
        real_scores = real_out

        z = Variable(torch.randn(num_img, z_dimension)).to(device)
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = loss(fake_out, fake_label)
        fake_scores = fake_out

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        z = Variable(torch.randn(num_img, z_dimension)).to(device)
        fake_img = G(z)
        output = D(fake_img)
        g_loss = loss(output, real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'
                  .format(epoch, num_epoch, d_loss.data, g_loss.data,
                          real_scores.data.mean(), fake_scores.data.mean()))
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './dc_img/fake_images-{}.png'.format(epoch+1))