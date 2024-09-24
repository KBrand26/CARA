import pytorch_lightning as pl

class ImageLogCallback(pl.Callback):
    def __init__(self):
        """
        Initialize the image logging callback.
        """
        self.epoch = -1
        self.first = True
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Logs reconstructions of the first 5 images in a validation batch to WandB.

        Args:
            trainer (pl.Trainer): The trainer that is being used to train the model.
            pl_module (pl.LightningModule): The model being trained.
            outputs (List): The outputs from the validation step.
            batch (Tensor): The batch used during the validation step.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
        if batch_idx == 0:
            self.epoch += 1
            x,_,_ = batch            
            images = [img for img in x[:5]]
            image_captions = [f"Original image {i}" for i in range(5)]
            recons = [rec for rec in outputs[1][:5]]
            rec_captions = [f"Reconstructed image {i}" for i in range(5)]
            
            pl_module.logger.log_image(key="Reconstructions", images=images+recons, caption=image_captions+rec_captions)