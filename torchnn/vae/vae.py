import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            *self._block(input_dim, 300),
            *self._block(300, 20),
            *self._block(300, 20),
            *self._block(20, 300),
            *self._block(300, input_dim),
        )
    
    def _block(self, in_features, out_features, batch_norm=0.8):
        return nn.Linear(in_features, out_features)
