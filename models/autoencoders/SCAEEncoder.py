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

class SCAEEncoder(pl.LightningModule):
    """
    Encoder portion of the SCAE that reduces the dimensionality of the given images.
    """
    def __init__(self, ld, filters=64, nconv=2, dense=False, flatten=False):
        """Initializes the encoder. 

        Args:
            ld (int): The dimension of the latent space.
            filters (int, optional): The number of filters to start with in first layer. Defaults to 64.
            nconv (int, optional): The number of convolutional blocks to use. Defaults to 2.
            dense (bool, optional): Flag that indicates whether dense layers should be used. Defaults to False.
            flatten (bool, optional): Flag that indicates whether the encoding should be flattened. Defaults to False.
        """
        super().__init__()
        # Save hyperparameters in case loading necessary.
        self.save_hyperparameters()
        
        layers = []
        # Create layers
        layers.append(
            nn.Conv2d(
                in_channels=1, # Monochrome
                out_channels=filters,
                kernel_size=3,
                padding="same"
            )
        )
        layers.append(nn.ReLU())
        for i in range(nconv):
            # Add convolutional blocks
            prev_filters = filters
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                ) 
            )
            layers.append(
                nn.Conv2d(
                    in_channels=prev_filters, # Previous amount of filters
                    out_channels=filters,
                    kernel_size=3,
                    padding="same"
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=3,
                    padding="same"
                )
            )
            layers.append(nn.ReLU())
        if dense:
            layers.append(
                nn.MaxPool2d(
                    kernel_size=4
                ) 
            )
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    in_features=20736,
                    out_features=ld
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels=filters,
                    out_channels=28,
                    kernel_size=3,
                    padding="same"
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.MaxPool2d(
                    kernel_size=3
                ) 
            )
            if flatten:
                layers.append(nn.Flatten())
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
    enc = SCAEEncoder(4032)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = enc.network.to(device)
    summary(model, (1, 150, 150))