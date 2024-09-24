import torch
import pytorch_lightning as pl
import numpy as np
import sys
# tell interpreter where to look
sys.path.insert(0,"data/datamodules")
from CleanDataModule import CleanDataModule
from model_constructor import construct_memscae, construct_scae, construct_bcae
import warnings
import wandb
import os
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import MinMaxScaler

def basic_mse(rec, og):
    """Calculates the mean squared error between two given images.

    Args:
        rec (ndarray): The reconstructed image.
        og (ndarray): The original image.

    Returns:
        float: The mean squared error between two images.
    """
    # Calculate the error
    diff = og-rec
    squared = np.square(diff)
    return squared.mean()

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

def train_model(model, train_loader, val_loader, loggers, callbacks, epochs=30):
    """Trains the given model using the given data.

    Args:
        model (LightningModule): The model to train.
        train_loader (DataLoader): The dataloader to use for training data.
        val_loader (DataLoader): The dataloader to use for validation data.
        loggers (List): The set of loggers to use during training.
        callbacks (List): The callbacks to use during training.
        epochs (int, optional): The maximum number of epochs to train the autoencoder. Defaults to 30.
    """
    trainer = pl.Trainer(
        accelerator="auto",
        deterministic=False,
        max_epochs=epochs,
        log_every_n_steps=10,
        logger=loggers,
        callbacks=callbacks
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

class ExperimentRunner():
    SCAE_LABEL = {
        True: "SCAE_",
        False: "BCAE_"
    }
    RESAMPLE_LABEL = {
        True: "resampling_",
        False: "no_resampling_"
    }
    MEMORY_LABEL = {
        True: "memory_",
        False: "no_memory_"
    }
    ENTROPY_LABEL = {
        True: "entropy_",
        False: "no_entropy_"
    }
    FILTER_LABEL = {
        True: "filtering_",
        False: "no_filtering_"
    }
    
    def __init__(self, runs, scae=True, resample=True, filtering=True, flatten=False, memory=False, save=False, epochs=30):
        """Initializes an instance of an experiment runner.

        Args:
            runs (int): Indicates how many runs to execute during experimentation.
            scae (bool, optional): A flag that indicates whether to use the SCAE architecture. Defaults to True.
            resample (bool, optional): A flag that indicates whether resampling should be applied. Defaults to True.
            filtering (bool, optional): A flag that indicates whether filtering should be applied to the data. Defaults to True.
            flatten (bool, optional): A flag that indicates whether to flatten encodings. Defaults to False.
            memory (bool, optional): A flag that indicates whether to use a memory unit. Defaults to False.
            save (bool, optional): A flag that indicates whether to save reconstructions. Defaults to False.
            epochs (int, optional): Indicates the number of epochs to train for each run. Defaults to 30.
        """
        self.runs = runs
        self.scae = scae
        self.epochs = epochs
        self.resample = resample
        self.filter = filtering
        self.flatten = flatten
        self.memory = memory
        self.save = save
    
    def setup_parameters(self, mem_size=100, ld=4032):
        """Sets up the normal and anomaly classes, the data module and the `latent dimension' of the autoencoder.

        Args:
            ld (int, optional): Indicates what the dimension of the encoding space should be. Defaults to 4032.
            mem_size (int, optional): What size should be used for the memory unit. Defaults to 100.
        """
        pl.seed_everything(42)
        self.mem_size = mem_size
        self.ld = ld
        self.anomaly_classes = [40, 50]
        self.normal_classes = [10, 20]
        self.mbcdm = CleanDataModule("data/FRGADB_Numpy/", self.normal_classes, self.anomaly_classes, 64, 42, \
            augment=True, filtering=self.filter, resample=self.resample)
    
    def print_run_start_info(self):
        """
        Simple utility function to announce the start of a run
        """
        print("=============================================================================================")
        print(f"Starting run {self.current_run}.")
        print("=============================================================================================")
    
    def execute_model_training(self, shrink=0.02, entropy=False, entropy_weight=0.0002):
        """This function is used to conduct model training

        Args:
            shrink (float, optional): Indicates what threshold to use if applying hard shrinkage to memscae models. 
                Defaults to 0.02.
            entropy (bool, optional): Indicates whether to add an entropy element to the loss function of memscae models.
                Defaults to False.
            entropy_weight (float, optional): The weight to use when adding an entropy element to the loss.
                Defaults to 0.0002.
        """
        train_loader = self.mbcdm.train_dataloader()
        val_loader = self.mbcdm.val_dataloader()
        
        # Construct the models with appropriate parameters
        if self.scae:
            if self.memory:
                good_lr = 0.00008
                self.ae, callbacks = construct_memscae(lr=good_lr, callbacks=True, flatten=self.flatten, mem_size=self.mem_size, shrink=shrink, entropy=entropy)
            else:
                good_lr = 0.00005
                self.ae, callbacks = construct_scae(self.ld, lr=good_lr, callbacks=True, flatten=self.flatten)
        else:
            good_lr = 0.0007
            self.ae, callbacks = construct_bcae(9, lr=good_lr, callbacks=True, deep=False, dense=False)

        # Determine the name of the run
        model = self.SCAE_LABEL[self.scae]
        resamp = self.RESAMPLE_LABEL[self.resample]
        filtering = self.FILTER_LABEL[self.filter]
        if self.memory:
            memory = self.MEMORY_LABEL[self.memory]
            entrop = self.ENTROPY_LABEL[entropy]
            self.group_name = f"{memory}size{self.mem_size}_shrink{shrink}_{entrop}{entropy_weight}_{resamp}{filtering}"
            self.run_name = f"{self.group_name}run{self.current_run}"
        else:
            self.group_name = f"{model}{resamp}{filtering}nn_"
            self.run_name = f"{self.group_name}run{self.current_run}"
        
        # Create loggers for experiment tracking
        csv_logger = pl.loggers.CSVLogger("lightning_logs", name=self.run_name)
        self.wandb_logger = pl.loggers.WandbLogger(project="CARA", group=self.group_name, name=self.run_name, \
            log_model=True, reinit=True)
        loggers = [self.wandb_logger, csv_logger]

        # Train the CAE
        train_model(self.ae, train_loader, val_loader, loggers, callbacks, epochs=self.epochs)
        
        if self.save:
            # Record outputs on training set if necessary
            self.ae.eval()
            
            self.images = np.array([])
            self.latents = np.array([])
            self.reconstructions = np.array([])
            self.true_labels = np.array([])
            self.labels = np.array([])
            
            train_loader = self.mbcdm.train_dataloader()
            first = True
            for batch in train_loader:
                batch_x, batch_y, batch_y_true = batch
                # Turn off gradient tracking for efficiency
                with torch.no_grad():
                    if self.memory:
                        out = self.ae(batch_x)
                        recon = out['recon']
                        attent = out['attention']
                    else:
                        latent, recon = self.ae(batch_x)
                    batch_x = batch_x.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    batch_y_true = batch_y_true.detach().cpu().numpy()
                    latent = latent.detach().cpu().numpy()
                    recon = recon.detach().cpu().numpy()

                    if first:
                        self.images = batch_x
                        self.latents = latent
                        self.reconstructions = recon
                        self.true_labels = batch_y_true
                        self.labels = batch_y
                        first = False
                    else:
                        self.images = np.concatenate([self.images, batch_x])
                        self.latents = np.concatenate([self.latents, latent])
                        self.reconstructions = np.concatenate([self.reconstructions, recon])
                        self.true_labels = np.concatenate([self.true_labels, batch_y_true])
                        self.labels = np.concatenate([self.labels, batch_y])
            
            os.makedirs(
                f"saved_ae_data/{self.run_name}/",
                exist_ok=True
            )
            np.save(f"saved_ae_data/{self.run_name}/train_images.npy", self.images)
            np.save(f"saved_ae_data/{self.run_name}/train_latents.npy", self.latents)
            np.save(f"saved_ae_data/{self.run_name}/train_recons.npy", self.reconstructions)
            np.save(f"saved_ae_data/{self.run_name}/train_true_labels.npy", self.true_labels)
            np.save(f"saved_ae_data/{self.run_name}/train_labels.npy", self.labels)
    
    def execute_anomaly_validation_loop(self):
        """
        This function conducts a run of the validation loop, including the validation anomalies
        """
        # Set model to eval mode to ensure correct validation and testing behaviour
        self.ae.eval()
        
        self.images = np.array([])
        self.latents = np.array([])
        self.reconstructions = np.array([])
        self.true_labels = np.array([])
        self.labels = np.array([])
        self.srcs = np.array([])
        
        full_val_loader = self.mbcdm.ano_val_dataloader()
        first = True
        for batch in full_val_loader:
            batch_x, batch_y, batch_y_true, batch_src = batch
            # Turn off gradient tracking for efficiency
            with torch.no_grad():
                if self.memory:
                    out = self.ae(batch_x)
                    recon = out['recon']
                    attent = out['attention']
                else:
                    latent, recon = self.ae(batch_x)
                batch_x = batch_x.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_y_true = batch_y_true.detach().cpu().numpy()
                batch_src  = np.array(batch_src)
                latent = latent.detach().cpu().numpy()
                recon = recon.detach().cpu().numpy()
                
                if first:
                    self.images = batch_x
                    self.latents = latent
                    self.reconstructions = recon
                    self.true_labels = batch_y_true
                    self.labels = batch_y
                    self.srcs = batch_src
                    first = False
                else:
                    self.images = np.concatenate([self.images, batch_x])
                    self.latents = np.concatenate([self.latents, latent])
                    self.reconstructions = np.concatenate([self.reconstructions, recon])
                    self.true_labels = np.concatenate([self.true_labels, batch_y_true])
                    self.labels = np.concatenate([self.labels, batch_y])
                    self.srcs = np.concatenate([self.srcs, batch_src])
        
        if self.save:
            os.makedirs(
                f"saved_ae_data/{self.run_name}/",
                exist_ok=True
            )
            np.save(f"saved_ae_data/{self.run_name}/val_images.npy", self.images)
            np.save(f"saved_ae_data/{self.run_name}/val_latents.npy", self.latents)
            np.save(f"saved_ae_data/{self.run_name}/val_recons.npy", self.reconstructions)
            np.save(f"saved_ae_data/{self.run_name}/val_true_labels.npy", self.true_labels)
            np.save(f"saved_ae_data/{self.run_name}/val_labels.npy", self.labels)
        
        # Record some basic metrics (assists with hyperparameter tuning)
        std_mses = []
        for i in range(len(self.images)):
            og = self.images[i]
            rec = self.reconstructions[i]
            std_mse = basic_mse(rec=rec, og=og)
            std_mses.append(std_mse)
        std_mses = np.array(std_mses)

        self.std_mse_scaler = MinMaxScaler()
        std_mses = self.std_mse_scaler.fit_transform(std_mses.reshape(-1, 1))
        
        thresh_min, thresh_max = std_mses.min(), std_mses.max()
        pos_thresholds = np.linspace(thresh_min, thresh_max, 100)
        pos_gmeans = []
        for thresh in pos_thresholds:
            std_mses_thresh = (std_mses > thresh).astype(int)
            pos_gmeans.append(geometric_mean_score(self.labels, std_mses_thresh, average="binary"))
        arg_best_gmean = np.argmax(pos_gmeans)
        self.ano_thresh = pos_thresholds[arg_best_gmean]
        
        val_pred_y = (std_mses > self.ano_thresh).astype(int)
        val_gmean = geometric_mean_score(self.labels, val_pred_y, average="binary")
        val_f1 = f1_score(self.labels, val_pred_y, average="binary", zero_division=0.0)
        val_prec = precision_score(self.labels, val_pred_y, average="binary", zero_division=0.0)
        val_rec = recall_score(self.labels, val_pred_y, average="binary", zero_division=0.0)
        val_f2 = fbeta_score(self.labels, val_pred_y, beta=2, average="binary", zero_division=0.0)
        
        self.wandb_logger.log_metrics({
            'Validation G-Mean': val_gmean,
            'Validation F1Score': val_f1,
            'Validation F2Score': val_f2,
            'Validation Precision': val_prec,
            'Validation Recall': val_rec,
        })
    
    def execute_anomaly_testing_loop(self):
        """
        This function handles the logic for a loop through the testing data.
        """
        # Set model to eval mode to ensure correct validation and testing behaviour
        self.ae.eval()
        
        self.images = np.array([])
        self.latents = np.array([])
        self.reconstructions = np.array([])
        self.true_labels = np.array([])
        self.labels = np.array([])
        self.srcs = np.array([])
        
        test_loader = self.mbcdm.test_dataloader()
        first = True
        for batch in test_loader:
            # Generate outputs
            batch_x, batch_y, batch_y_true, batch_src = batch
            # Turn off gradient tracking for efficiency
            with torch.no_grad():
                if self.memory:
                    out = self.ae(batch_x)
                    recon = out['recon']
                    attent = out['attention']
                else:
                    latent, recon = self.ae(batch_x)
                batch_x = batch_x.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_y_true = batch_y_true.detach().cpu().numpy()
                batch_src  = np.array(batch_src)
                latent = latent.detach().cpu().numpy()
                recon = recon.detach().cpu().numpy()
                
                if first:
                    self.images = batch_x
                    self.latents = latent
                    self.reconstructions = recon
                    self.true_labels = batch_y_true
                    self.labels = batch_y
                    self.srcs = batch_src
                    first = False
                else:
                    self.images = np.concatenate([self.images, batch_x])
                    self.latents = np.concatenate([self.latents, latent])
                    self.reconstructions = np.concatenate([self.reconstructions, recon])
                    self.true_labels = np.concatenate([self.true_labels, batch_y_true])
                    self.labels = np.concatenate([self.labels, batch_y])
                    self.srcs = np.concatenate([self.srcs, batch_src])
        
        if self.save:
            os.makedirs(
                f"saved_ae_data/{self.run_name}/",
                exist_ok=True
            )
            np.save(f"saved_ae_data/{self.run_name}/test_images.npy", self.images)
            np.save(f"saved_ae_data/{self.run_name}/test_latents.npy", self.latents)
            np.save(f"saved_ae_data/{self.run_name}/test_recons.npy", self.reconstructions)
            np.save(f"saved_ae_data/{self.run_name}/test_true_labels.npy", self.true_labels)
            np.save(f"saved_ae_data/{self.run_name}/test_labels.npy", self.labels)

        # Determine how well autoencoder can separate anomalies using only standard MSE. Only used as a first
        # impression of model performance in anomaly detection.
        std_mses = []
        for i in range(len(self.images)):
            og = self.images[i]
            rec = self.reconstructions[i]
            std_mse = basic_mse(rec=rec, og=og)
            std_mses.append(std_mse)
        std_mses = np.array(std_mses)

        # Use validation data scaler to normalize MSEs
        std_mses = self.std_mse_scaler.transform(std_mses.reshape(-1, 1))
        
        test_pred_y = (std_mses > self.ano_thresh).astype(int)
        test_gmean = geometric_mean_score(self.labels, test_pred_y, average="binary")
        test_f1 = f1_score(self.labels, test_pred_y, average="binary", zero_division=0.0)
        test_prec = precision_score(self.labels, test_pred_y, average="binary", zero_division=0.0)
        test_rec = recall_score(self.labels, test_pred_y, average="binary", zero_division=0.0)
        test_f2 = fbeta_score(self.labels, test_pred_y, beta=2, average="binary", zero_division=0.0)
        
        self.wandb_logger.log_metrics({
            'Testing G-Mean': test_gmean,
            'Testing F1Score': test_f1,
            'Testing F2Score': test_f2,
            'Testing Precision': test_prec,
            'Testing Recall': test_rec,
        })

    def execute_experiment(self, shrink=0.02, entropy=False, mem_size=100):
        """This function serves as the main driver for the execution of the experiment.

        Args:
            shrink (float, optional): Indicates the threshold to use if applying hard shrinkage.
                Defaults to 0.02.
            entropy (bool, optional): Indicates whether to use entropy. Defaults to False.
            mem_size (int, optional): Indicates the size of the memory unit when using MemSCAE. Defaults to 100.
        """
        self.setup_parameters(mem_size=mem_size)
        
        for r in range(1, self.runs+1):
            self.execute_run(r, shrink=shrink, entropy=entropy)
        
            if r != self.runs:
                # Setup for next run
                self.mbcdm.create_next_train_val_sets()
                del self.ae
                print(f"Completed run {r}")
                self.wandb_logger.finalize("success")
                wandb.finish()
            else:
                print("Experiment concluded")
                self.wandb_logger.finalize("success")
                wandb.finish()

    def execute_run(self, run, shrink=0.02, entropy=False):
        """Executes a single experimental run

        Args:
            run (int): Indicates the run number.
            shrink (float, optional): The threshold to use if using hard shrinkage. Defaults to 0.02.
            entropy (bool, optional): Indicates whether to use an entropy component in the loss function of 
                MemSCAE. Defaults to False.
        """
        self.current_run = run
        
        self.print_run_start_info()
        self.execute_model_training(shrink=shrink, entropy=entropy)
        self.execute_anomaly_validation_loop()
        self.execute_anomaly_testing_loop()

if __name__ == "__main__":    
    # Run MemSCAE with sparsity
    # runner = ExperimentRunner(runs=10, scae=True, resample=True, filtering=True, flatten=True, memory=True, save=True, epochs=30)
    # runner.execute_experiment(shrink=0.002, entropy=False, mem_size=500)
    
    # # Run SCAE experiment
    # runner = ExperimentRunner(runs=10, scae=True, flatten=True, memory=False, save=True, epochs=30)
    # runner.execute_experiment()
    
    # Run BCAE experiment
    runner = ExperimentRunner(runs=10, scae=False, memory=False, save=True, epochs=30)
    runner.execute_experiment()