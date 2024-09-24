import pytorch_lightning as pl
import sys
# tell interpreter where to look
sys.path.insert(0, "callbacks/")
sys.path.insert(0, "models/autoencoders/")
from Autoencoder import Autoencoder
from MemoryAutoencoder import MemoryAutoencoder
from SCAEDecoder import SCAEDecoder
from SCAEEncoder import SCAEEncoder
from BCAEEncoder import BCAEEncoder
from BCAEDecoder import BCAEDecoder
from ImageLogCallback import ImageLogCallback

def construct_scae(ld, lr = 0.000001, upsample = True, callbacks=False, dense=False, flatten=False):
    """This function is used to construct a SCAE, along with it's corresponding callbacks.

    Args:
        ld (int): Indicates the size of the "latent" dimension for this autoencoder.
        lr (float, optional): The learning rate to use for this model. Defaults to 0.000001.
        upsample (bool, optional): Whether to use upsampling in the model. Defaults to True.
        callbacks (bool, optional): Indicates whether to also construct the callbacks for this model
            or not. Defaults to False.
        dense (bool, optional): Indicates whether to make use of dense layers. Defaults to False.
        flatten (bool, optional): Indicates whether to flatten the encodings. Defaults to False.

    Returns:
        Tuple: Returns a tuple containing the model and callbacks for that model.
    """
    if callbacks:
        escb = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=5)
        cpcb = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
        ilcb = ImageLogCallback()
        callbacks = [escb, cpcb, ilcb]
    else:
        callbacks = []
        
    enc = SCAEEncoder(ld, dense=dense, flatten=flatten)
    dec = SCAEDecoder(ld, upsample=upsample, dense=dense, flatten=flatten)
    ae = Autoencoder(enc, dec, lr=lr)
    
    return ae, callbacks

def construct_memscae(lr = 0.00005, callbacks=True, flatten=True, mem_size=100, shrink=0.02, entropy=False, \
    entropy_weight=0.0002):
    """This function constructs a MemSCAE using the SCAE encoder and decoder.

    Args:
        lr (float, optional): The learning rate to use. Defaults to 0.00005.
        callbacks (bool, optional): Indicates whether to also construct the callbacks for this model
            or not. Defaults to True.
        flatten (bool, optional): Indicates whether to flatten the encodings. Defaults to True.
        mem_size (int, optional): The size to use for the memory unit. Defaults to 100.
        shrink (float, optional): The threshold to use for hard shrinkage. Defaults to 0.0.
        entropy (bool, optional): Whether to make use of an entropy loss for regularization. Defaults to False.
        entropy_weight (bool, optional): The weight to use when adding the entropy loss. Will only be used if 
            entropy is True. Defaults to 0.0002.

    Returns:
        Tuple: Returns a tuple containing the model and callbacks for that model.
    """
    if callbacks:
        escb = pl.callbacks.early_stopping.EarlyStopping(monitor="val_recon_loss", mode="min", patience=5)
        cpcb = pl.callbacks.ModelCheckpoint(monitor="val_recon_loss", mode="min")
        ilcb = ImageLogCallback()
        callbacks = [escb, cpcb, ilcb]
    else:
        callbacks = []
        
    enc = SCAEEncoder(4032, flatten=flatten)
    dec = SCAEDecoder(4032, upsample=True, flatten=flatten)
    if entropy:
        ae = MemoryAutoencoder(enc, dec, lr=lr, mem_size=mem_size, shrink_thresh=shrink, entropy=entropy, \
            entropy_weight=entropy_weight)
    else:
        ae = MemoryAutoencoder(enc, dec, lr=lr, mem_size=mem_size, shrink_thresh=shrink, entropy=entropy, \
            entropy_weight=0)
    
    return ae, callbacks

def construct_bcae(reduc_factor, lr = 0.000001, upsample = True, callbacks = False, deep = False, dense = False):
    """This function is used to construct a BCAE, along with it's corresponding callbacks.

    Args:
        reduc_factor (int): The factor with which to reduce the dimensionality of the images after 
            the last convolutional layer.
        lr (float, optional): The learning rate to use for this model during training. Defaults to 0.000001.
        upsample (bool, optional): Whether to use upsampling in this model. Defaults to True.
        callbacks (bool, optional): Indicates whether to also construct the callbacks for this model.
            Defaults to False.
        deep (bool, optional): Indicates whether to use a deep variant of the autoencoder.
            Defaults to False.
        dense (bool, optional): Indicates whether to dense layers to the autoencoder.
            Defaults to False.

    Returns:
        Tuple: Returns a tuple containing the model as well as the callbacks necessary for training the model.
    """
    if callbacks:
        escb = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=10)
        cpcb = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
        ilcb = ImageLogCallback()
        callbacks = [escb, cpcb, ilcb]
    else:
        callbacks = []

    enc = BCAEEncoder(reduc_factor, deep=deep, dense=dense)
    dec = BCAEDecoder(reduc_factor, upsample=upsample, deep=deep, dense=dense)
    ae = Autoencoder(enc, dec, lr=lr)
    
    return ae, callbacks