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
sys.path.insert(0,"data/datamodules")
sys.path.insert(0, "callbacks/")
from CleanDataModule import CleanDataModule
from ImageLogCallback import ImageLogCallback
import warnings

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

class Autoencoder(pl.LightningModule):
    LOSSES = {
        "MSE": mse_loss,
    }
    
    def __init__(self, encoder, decoder, lr=1e-3, loss="MSE"):
        """Initializes the autoencoder

        Args:
            encoder (Object): An encoder to use in this autoencoder.
            decoder (Object): A decoder to use in this autoencoder.
            lr (float): The learning rate to use to tune the model.
            loss (str, optional): Indicates the loss to use when training this autoencoder. Defaults to "MSE".
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        """Pass the inputs from the batch through the autoencoder

        Args:
            x (Tensor): Batch of input images.

        Returns:
            Tuple: A tuple containing the latent representations and reconstructed images corresponding to the given batch.
        """
        latent = self.encoder(x)
        x = self.decoder(latent)
        return latent, x
    
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

    def training_step(self, train_batch, batch_idx):
        """Defines what should happen during each step during training

        Args:
            train_batch (Tensor): A batch of images to pass through the model.
            batch_idx (int): The index of the given batch

        Returns:
            float: The loss for the given batch
        """
        img, _, _ = train_batch
        _, recon = self.forward(img)
        loss = self.reconstruction_loss(img=img, recon=recon)
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
        latent, recon = self.forward(img)
        loss = self.reconstruction_loss(img=img, recon=recon)
        self.log('val_loss', loss)
        
        return latent, recon
    
if __name__=="__main__":
    # Check whether class creates model as expected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = SCAEEncoder(4032).to(device)
    dec = SCAEDecoder(4032, upsample=True).to(device)
    ae = Autoencoder(enc, dec).to(device)
    summary(ae, (1, 150, 150))