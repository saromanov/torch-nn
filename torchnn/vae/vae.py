import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
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
    
    def forward(self, x):
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        return self._decode(z), mu, logvar
