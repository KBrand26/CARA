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
        
class SCAEDecoder(pl.LightningModule):
    """
    Decoder portion of the SCAE that reconstructs images from their compressed representations.
    """
    def __init__(self, ld, upsample=False, filters=64, nconv=2, unf_size=9, dense=False, flatten=False):
        """Initializes the decoder. 

        Args:
            ld (int): The dimension of the latent space.
            upsample (bool, optional): Flag that indicates whether to use nearest neighbour upsampling
              layers. Defaults to False.
            filters (int, optional): The number of filters to start with in first convolutional layer.
              Defaults to 256.
            nconv (int, optional): The number of convolutional blocks to use. Defaults to 2.
            unf_size (int, optional): The size of the grid required after unflattening. Defaults to 18.
            dense (bool, optional): A flag that indicates whether to make use of dense layers. Defaults to False.
            flatten (bool, optional): A flag that indicates whether the encoding should be flattened. Defaults to False
        """
        super().__init__()
        # Save hyperparameters in case loading necessary.
        self.save_hyperparameters()
        
        layers = []
        if dense:
            layers.append(
                nn.Linear(
                    in_features=ld,
                    out_features=20736
                )
            )
            layers.append(nn.ELU())
            layers.append(nn.Unflatten(dim=1, unflattened_size=(filters, unf_size, unf_size)))
        elif upsample:
            if flatten:
                layers.append(nn.Unflatten(dim=1, unflattened_size=(28, 12, 12)))
            layers.append(
                nn.Upsample(
                    scale_factor=3
                )
            )
            unf_size = 36
            layers.append(
                nn.Conv2d(
                    in_channels=28,
                    out_channels=filters,
                    kernel_size=3,
                    padding="same"
                )
            )
            layers.append(nn.ReLU())
        elif flatten:
            layers.append(nn.Unflatten(dim=1, unflattened_size=(28, 12, 12)))
        prev_filters = filters
        for i in range(nconv):
            # Add "deconvolutional" blocks
            unf_size = (unf_size*2) + 1
            if upsample:
                layers.append(
                    nn.Upsample(
                        size=(unf_size, unf_size)
                    )
                )
                layers.append(
                    nn.Conv2d(
                        in_channels=prev_filters,
                        out_channels=filters,
                        kernel_size=3,
                        padding="same"
                    )
                )
                layers.append(nn.ReLU())
            else:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=prev_filters,
                        out_channels=filters,
                        kernel_size=3,
                        stride=2,
                        padding=2
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
            prev_filters = filters
        if upsample:
            layers.append(
                nn.Upsample(
                    size=(150, 150)
                )
            )
            layers.append(
                nn.Conv2d(
                    in_channels=prev_filters,
                    out_channels=filters,
                    kernel_size=3,
                    padding="same"
                )
            )
            layers.append(nn.ReLU())
        else:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=prev_filters,
                    out_channels=filters,
                    kernel_size=8,
                    stride=2,
                    padding=3
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Conv2d(
                in_channels=filters,
                out_channels=1,
                kernel_size=3,
                padding="same"
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
    dec = SCAEDecoder(4032, upsample=True, flatten=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dec.network.to(device)
    summary(model, (28, 12, 12))