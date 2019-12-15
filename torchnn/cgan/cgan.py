import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, num_classes, dims):
        super(Generator, self).__init__()
        self.labels_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            *block(dims + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        gen_input = torch.cat((self.labels_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img