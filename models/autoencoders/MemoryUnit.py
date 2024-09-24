import torch
from torch import nn, optim
import pytorch_lightning as pl
import math
from torch.nn import functional as F

# This class is inspired heavily by the the original implementation done by Gong et al (2019)
# It is publicly available at: https://github.com/donggong1/memae-anomaly-detection/blob/master/models/memory_module.py
class MemoryUnit(pl.LightningModule):
    def __init__(self, mem_size, latent_size=4032, shrink_thresh=0.02):
        """Initializes the memory unit.

        Args:
            mem_size (int): Indicates how many encodings should be kept in memory.
            latent_size (int, optional): Indicates how large the latent space is. Defaults to 4032.
            shrink_thresh (float, optional): Indicates the threshold to use when applying hard shrinkage. Defaults to 0.02.
        """
        super().__init__()
        self.md = mem_size
        self.ld = latent_size
        self.memory = nn.Parameter(torch.Tensor(self.md, self.ld)) # Mem size (N) x Latent size (Z)
        self.thresh = shrink_thresh
        
        self.init_weights()
        
    def init_weights(self):
        """
        Initializes the parameters of the memory unit
        """
        # Based on the memory unit initialization from Gong et al (2019)
        sigma = 1./math.sqrt(self.memory.size(1))
        self.memory.data.uniform_(-sigma, sigma)
        
    def relu_shrink(self, attent, eps=1e-12):
        """Applies hard shrinkage to the attention weights to ensure sparse weights. This is done using the
          ReLU operator to ensure that the backwards pass is straightforward.

        Args:
            attent (Tensor): The attention weights to apply shrinkage to.
            eps (float, optional): An epsilon value to avoid division by zero. Defaults to 1e-12.

        Returns:
            Tensor: The attention weights after applying shrinkage.
        """
        num = F.relu(attent-self.thresh)*attent
        denom = torch.abs(attent - self.thresh) + eps
        return num/denom

    def forward(self, x):
        """Conduct a forward pass of the memory unit.

        Args:
            x (Tensor): The latent input to pass through the unit.

        Returns:
            dict: A dict containing the new encoding as well as the attention weights from memory.
        """
        attent = F.linear(x, self.memory) # Calculate similarity -> xW^T, (batch size (B) x Z)*(ZxN) -> BxN
        attent = F.softmax(attent, dim=1) # Ensure that dimension one sums to 1 (memory weights sum to one)
        
        if self.thresh > 0:
            # Threshold attention weights to reduce likelihood of complex reconstructions
            attent = self.relu_shrink(attent)
            # Normalize attention weights again to ensure sum to one
            attent = F.normalize(attent, p=1, dim=1)
        # Get new latent representation from memory
        memory_transpose = self.memory.permute(1, 0) # NxZ -> ZxN
        zhat = F.linear(attent, memory_transpose) # BxN * NxZ -> BxZ
        return {
            'zhat': zhat,
            'att': attent
        }