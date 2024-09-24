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

class BCAEEncoder(pl.LightningModule):
    """
    Encoder portion of the autoencoder that reduces the dimensionality of the given images.
    Heavily inspired by F.L. Ventura's masters thesis:
    Prospecting for enigmatic radio sources with autoencoders : a novel approach
    """
    def __init__(self, scale=9, deep=False, dense=False):
        """Initializes the encoder. 

        Args:
            scale (int): The factor with which to reduce the data in the max pooling layer. Defaults to 9.
            deep (bool): Flag indicating whether a deep variant of the network should be used. Defaults to False.
            dense (bool): Flag indicating whether dense layers should be added to the network. Defaults to False.
        """
        super().__init__()
        # Save hyperparameters in case loading necessary.
        self.save_hyperparameters()
        
        layers = []
        # Create layers
        layers.append(
            nn.Conv2d(
                in_channels=1, # Monochrome
                out_channels=16,
                kernel_size=3,
            )
        )
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
            )
        )
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
            )
        )
        layers.append(nn.ReLU())
        if deep:
            layers.append(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.MaxPool2d(
                kernel_size=scale
            )
        )
        if dense:
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    in_features=5184,
                    out_features=4096
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
    enc = BCAEEncoder(9, deep=False, dense=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = enc.network.to(device)
    summary(model, (1, 150, 150))