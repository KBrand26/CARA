import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchsummary import summary

def initialize_weights(layer):
    """Initializes the weights of a given layer

    Args:
        layer (object): The PyTorch layer to initialize.
    """
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
        
class BCAEDecoder(pl.LightningModule):
    """
    Decoder portion of the autoencoder that reconstructs images from their compressed representations.
    Heavily inspired by F.L. Ventura's masters thesis:
    Prospecting for enigmatic radio sources with autoencoders : a novel approach
    """
    def __init__(self, scale=9, upsample=True, deep=False, dense=False):
        """Initializes the decoder. 

        Args:
            scale (int): The factor to use when upsampling the encoded data. Defaults to 9.
            upsample (bool, optional): Flag that indicates whether to use nearest neighbour upsampling
              layers. Defaults to True.
            deep (bool, optional): Flag that indicates whether to use a deep variant of the network.
                Defaults to False.
            dense (bool, optional): Flag that indicates whether to use dense layers in the network.
                Defaults to False.
        """
        super().__init__()
        # Save hyperparameters in case loading necessary.
        self.save_hyperparameters()
        
        layers = []
        if dense:
            layers.append(
                nn.Linear(
                    in_features=4096,
                    out_features=5184
                )
            )
            layers.append(nn.ELU())
            layers.append(nn.Unflatten(dim=1, unflattened_size=(16, 18, 18)))
        layers.append(
            nn.Upsample(
                scale_factor=scale
            )
        )
        if deep:
            layers.append(
                nn.ZeroPad2d(
                    padding=(1, 0, 1, 0)
                )
            )
        layers.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding="same"
            )
        )
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=2
            )
        )
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=2
            )
        )
        layers.append(nn.ReLU())
        if deep:
            layers.append(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    padding=3
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    padding=3
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                padding=2
            )
        )
        
        # Create model containing the constructed layers
        self.network = nn.Sequential(*layers)
        # Initialize weights
        self.network.apply(initialize_weights)
    
    def forward(self, x):
        """Executes a forward pass of the given data through the model.

        Args:
            x (Tensor): The batch to pass through the model.

        Returns:
            Tensor: The result produced by the network for the given batch.
        """
        return self.network(x)
        
if __name__ == "__main__":
    # Check whether class creates model as expected
    dec = BCAEDecoder(9, upsample=True, deep=False, dense=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dec.network.to(device)
    summary(model, (4096, ))