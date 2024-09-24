import os
import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchsummary import summary
from SCAEDecoder import SCAEDecoder
from SCAEEncoder import SCAEEncoder
from BCAEEncoder import BCAEEncoder
from BCAEDecoder import BCAEDecoder
import sys
# tell interpreter where to look
sys.path.insert(0,"data/datamodules/")
sys.path.insert(0, "callbacks/")
from CleanDataModule import CleanDataModule
from ImageLogCallback import ImageLogCallback
import warnings
from MemoryUnit import MemoryUnit
import numpy as np
from matplotlib import pyplot as plt

# Mute unnecessary warning
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

def mse_loss(recon, img):
    """Calculates and returns the mean squared error between the reconstructed images and the originals

    Args:
        recon (tensor): A tensor containing the reconstructed images
        img (tensor): A tensor containing the original images

    Returns:
        float: The mean MSE over the given batch
    """
    return nn.MSELoss()(recon, img)

class MemoryAutoencoder(pl.LightningModule):
    """
    Based on the MemAE model from Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for
    Unsupervised Anomaly Detection, Gong et al, 2019.
    """
    LOSSES = {
        "MSE": mse_loss,
    }
    
    def __init__(self, encoder, decoder, lr=1e-3, loss="MSE", mem_size=100, shrink_thresh=0.02, entropy=False, entropy_weight=0.0002):
        """Initializes the autoencoder

        Args:
            encoder (Object): An encoder to use in this autoencoder.
            decoder (Object): A decoder to use in this autoencoder.
            lr (float): The learning rate to use to tune the model.
            loss (str, optional): Indicates the loss to use when training this autoencoder. Defaults to "MSE".
            mem_size (int, optional): The size of the memory unit. Defaults to 100.
            shrink_thresh (float, optional): The threshold to use when shrinking. Defaults to 0.02.
            entropy (bool, optional): Indicates whether to add an entropy loss to regularize attention weights. Defaults to False.
            entropy_weight (bool, optional): The weight to use when adding the entropy loss. Will only be used if entropy is True. Defaults to 0.0002.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.encoder = encoder
        self.mem_unit = MemoryUnit(mem_size=mem_size, shrink_thresh=shrink_thresh)
        self.decoder = decoder
        
    def forward(self, x):
        """Pass the inputs from the batch through the autoencoder

        Args:
            x (Tensor): Batch of input images.

        Returns:
            dict: A dictionary containing the reconstructed image and the attention weights from the memory unit.
        """
        enc = self.encoder(x)
        mem_out = self.mem_unit(enc)
        mem_enc = mem_out['zhat']
        att = mem_out['att']
        x = self.decoder(mem_enc)
        return {
            'recon': x,
            'attention': att
        }
    
    def configure_optimizers(self):
        """Initializes the optimizer to use for this autoencoder.

        Returns:
            Object: The initialized optimizer.
        """
        optimizer = optim.NAdam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def reconstruction_loss(self, img, recon):
        """Calculates the reconstruction loss between the given batch and the corresponding reconstructions.

        Args:
            img (Tensor): The original images in the batch.
            recon (Tensor): The reconstructions produced by the autoencoder for the given batch.

        Returns:
            float: The reconstruction loss value.
        """
        loss = self.LOSSES.get(self.hparams.loss)
        return loss(recon=recon, img=img)
    
    def entropy_loss(self, att):
        """ Calculates the entropy loss as described in Gong et al, 2019 to regularize sparsity of the attention weights. Based on the implementation at
        https://github.com/donggong1/memae-anomaly-detection/blob/master/models/entropy_loss.py.
        
        Args:
            att (Tensor): The attention matrix returned by the memory unit.

        Returns:
            float: The entropy loss value.
        """
        loss = att * torch.log(att + 1e-15) # Add small value to prevent log(0)
        loss = -1.0 * loss.sum(dim=1)
        loss = loss.mean()
        return loss
    
    def visualize_memory(self, run):
        """This function is used to visualize what the elements in memory would look like if decoded.

        Args:
            run (int): Indicates what the current run number is.
        """
        with torch.no_grad():
            r = np.random.randint(0, self.hparams.mem_size, 5)
            cur_latent = self.mem_unit.memory[r]
            x = self.decoder(cur_latent).detach().cpu().numpy()
            
            os.makedirs("memory_vis/", exist_ok=True)
            for i in range(5):
                plt.imshow(x[i][0], cmap='gray')
                plt.savefig(f"memory_vis/run{run}_memory{i}.png")
                plt.show()

    def training_step(self, train_batch, batch_idx):
        """Defines what should happen during each step during training

        Args:
            train_batch (Tensor): A batch of images to pass through the model.
            batch_idx (int): The index of the given batch

        Returns:
            float: The loss for the given batch
        """
        img, _, _ = train_batch
        out = self.forward(img)
        recon = out["recon"]
        attent = out["attention"]
        recon_loss = self.reconstruction_loss(img=img, recon=recon)
        self.log('train_recon_loss', recon_loss)
        if self.hparams.entropy:
            entropy_loss = self.entropy_loss(att=attent)
            loss = recon_loss + self.hparams.entropy_weight*entropy_loss
            self.log('train_entropy_loss', entropy_loss)
            self.log('train_loss', loss)
        else:
            loss = recon_loss
            self.log('train_loss', loss)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        """Defines what should happen during each step during validation.

        Args:
            val_batch (Tensor): A batch of images to use for validation of the model.
            batch_idx (int): The index of the given batch.

        Returns:
            Tensor: The reconstructed images produced during validation.
        """
        img, _, _ = val_batch
        out = self.forward(img)
        recon = out['recon']
        attent = out["attention"]
        
        recon_loss = self.reconstruction_loss(img=img, recon=recon)
        self.log('val_recon_loss', recon_loss)        
        
        if self.hparams.entropy:
            entropy_loss = self.entropy_loss(att=attent)
            loss = recon_loss + self.hparams.entropy_weight*entropy_loss
            self.log('val_entropy_loss', entropy_loss)
            self.log('val_loss', loss)
        else:
            loss = recon_loss
            self.log('val_loss', loss)
        
        return recon