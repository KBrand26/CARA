from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
import numpy as np
from time import time
from imblearn.metrics import geometric_mean_score

class LocalOutlierFactorWrapper():
    # Hyperparameter combinations selected using validation G-Mean sweeps
    MEMSCAE_METRICS_PARAMETERS = {
        "n_neighbors" : 15,
        "metric" : "euclidean",
        "contamination" : 0.15,
    }
    BCAE_METRICS_PARAMETERS = {
        "n_neighbors" : 29,
        "metric" : "cosine",
        "contamination" : 0.31,
    }
    SCAE_METRICS_PARAMETERS = {
        "n_neighbors" : 25,
        "metric" : "manhattan",
        "contamination" : 0.19,
    }
    PCA_PARAMETERS = {
        "n_neighbors" : 15,
        "metric" : "manhattan",
        "contamination" : 0.2,
    }
    
    SWEEP_PARAMETERS = {
        "memscae_metrics": {
            "n_neighbors": {
                "values": [11, 13, 15, 17, 19],
            },
            "metric": {
                "values": ["euclidean", "cosine", "manhattan"],
            },
            "contamination": {
                "values": [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
            },
        },
        "scae_metrics": {
            "n_neighbors": {
                "values": [17, 19, 21, 23, 25],
            },
            "metric": {
                "values": ["euclidean", "cosine", "manhattan"],
            },
            "contamination": {
                "values": [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
            },
        },
        "bcae_metrics": {
            "n_neighbors": {
                "values": [21, 23, 25, 27, 29],
            },
            "metric": {
                "values": ["euclidean", "cosine", "manhattan"],
            },
            "contamination": {
                "values": [0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35]
            },
        },
        "pca": {
            "n_neighbors": {
                "values": [9, 11, 13, 15, 17, 19, 21],
            },
            "metric": {
                "values": ["euclidean", "cosine", "manhattan"],
            },
            "contamination": {
                "values": [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
            },
        },
    }

    novelty = True
    n_jobs = 8
    
    def __init__(self, mode):
        """Initialize the Local Outlier Factor wrapper.

        Args:
            mode (str): Indicates what type of data the model will be receiving.
        """
        self.mode = mode
    
    def set_default_model(self):
        """
        Create a local outlier factor model that uses the default parameters.
        """
        self.model = LocalOutlierFactor(novelty=self.novelty, n_jobs=self.n_jobs)
    
    def set_tuned_model(self):
        """
        Create a local outlier factor model using the tuned hyperparameters.
        """
        if self.mode == "pca":
            hparams = self.PCA_PARAMETERS
        elif self.mode == "memscae_metrics":
            hparams = self.MEMSCAE_METRICS_PARAMETERS
        elif self.mode == "scae_metrics":
            hparams = self.SCAE_METRICS_PARAMETERS
        elif self.mode == "bcae_metrics":
            hparams = self.BCAE_METRICS_PARAMETERS
        self.model = LocalOutlierFactor(
            n_neighbors=hparams["n_neighbors"],
            metric=hparams["metric"],
            contamination=hparams["contamination"],
            novelty=self.novelty,
            n_jobs=self.n_jobs
        )
    
    def set_model_params(self, n_neighbors, metric, contamination):
        """Create a local outlier factor model using the given parameters.

        Args:
            n_neighbors (int): The number of neighbors to consider when calculating the densities.
            metric (String): The metric to use when calculating the distances between samples.
            contamination (float): The percentage of samples to consider as contaminated.
        """
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            metric=metric,
            contamination=contamination,
            novelty=self.novelty,
            n_jobs=self.n_jobs
        )
    
    def set_sweep_data_function(self, data_func):
        """Sets up the function that is used to get the next data splits during sweeps.

        Args:
            data_func (func): The function to call when setting up the data splits for a hparam sweep.
        """
        self.data_func = data_func
    
    def cleanup_sweep(self):
        """
        Cleans up the data function and model after a sweep to ensure that the states are reset.
        """
        del self.data_func, self.model

    def fit_model(self, train_X, train_y, val_X, val_y):
        """Fit the local outlier factor model on the given data.

        Args:
            train_X (ndarray): The training data.
            train_y (ndarray): The labels corresponding to the training data.
            val_X (ndarray): The validation data. Not used for this model.
            val_y (ndarray): The labels corresponding to the validation data. Not used for this model.
        """
        # Remove anomalies, because LOF doesn't use them during training
        # (Later tests showed that keeping them did not matter either,
        # it only affected the value of the contamination hyperparameter)
        norm_idxs = train_y == 0
        self.model.fit(train_X[norm_idxs])
    
    def get_predictions(self, X):
        """Generate predictions for the given samples.

        Args:
            X (ndarray): The samples for which to generate anomaly predictions.

        Returns:
            ndarray: The predictions corresponding to the given samples.
        """
        pred = self.model.predict(X)
        pred[pred == 1] = 0
        pred[pred == -1] = 1
        return pred
    
    def get_proba_predictions(self, X):
        """
        This function is added to fit the expected behaviour, but it is not used.
        """
        return None
    
    def conduct_wandb_sweep(self, name, runs, first_sweep=True):
        """Conducts a hyperparameter sweep using Weights and Biases.

        Args:
            name (String): The name to use for the sweep.
            runs (int): The number of parameter combinations to test in the sweep.
            first_sweep (bool, optional): Indicates whether this is the first sweep or the finetuned sweep. Defaults to True.
        """
        self.get_sweep_config(name, first_sweep)
        sweep = wandb.sweep(self.config, project="CARAShallow")
        wandb.agent(sweep, self.sweep_fn, count=runs)
    
    def get_sweep_config(self, name, first_sweep=True):
        """Generates the configuration file for the hyperparameter sweep.

        Args:
            name (String): The name to use for the sweep.
            first_sweep (bool, optional): Indicates whether this is the first sweep or the finetuned sweep. Defaults to True.
        """
        if first_sweep:
            parameters = {
                "n_neighbors": {
                    "values": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
                },
                "metric": {
                    "values": ["euclidean", "cosine", "manhattan"],
                },
                "contamination": {
                    "values": ["auto", 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                },
            }
        else:
            parameters = self.SWEEP_PARAMETERS[self.mode]

        self.config = {
            'name': name,
            'method': 'bayes',
            'metric': {
                'name': 'Validation G-Mean',
                'goal': 'maximize',
            },
            'parameters': parameters
        }
    
    def sweep_fn(self, config=None):
        """This function is used by the hyperparameter sweep agents to test various hyperparameter combos.

        Args:
            config (dict, optional): The config to use for the current run. Defaults to None.
        """
        with wandb.init(config=config):
            # Get parameters from wandb agent
            config = wandb.config
            
            train_f1s = []
            train_gmeans = []
            train_precs = []
            train_recs = []
            val_f1s = []
            val_gmeans = []
            val_precs = []
            val_recs = []
            # Conduct 10 repetitions to ensure stable results
            for r in range(10):
                # Get data for run
                train_X, train_y, val_X, val_y, _, _, _, _ = self.data_func()
                
                # Set model parameters
                self.set_model_params(
                    **config
                )

                # Train model
                self.fit_model(train_X, train_y, None, None)
                train_y_pred = self.get_predictions(train_X)
                val_y_pred = self.get_predictions(val_X)

                train_prec = precision_score(train_y, train_y_pred, average='binary', zero_division=0.0)
                train_recall = recall_score(train_y, train_y_pred, average='binary', zero_division=0.0)
                train_f1 = f1_score(train_y, train_y_pred, average='binary', zero_division=0.0)
                train_gmean = geometric_mean_score(train_y, train_y_pred, average="binary")
                
                train_f1s.append(train_f1)
                train_gmeans.append(train_gmean)
                train_precs.append(train_prec)
                train_recs.append(train_recall)

                val_prec = precision_score(val_y, val_y_pred, average='binary', zero_division=0.0)
                val_recall = recall_score(val_y, val_y_pred, average='binary', zero_division=0.0)
                val_f1 = f1_score(val_y, val_y_pred, average='binary', zero_division=0.0)
                val_gmean = geometric_mean_score(val_y, val_y_pred, average="binary")
                
                val_f1s.append(val_f1)
                val_gmeans.append(val_gmean)
                val_precs.append(val_prec)
                val_recs.append(val_recall)
                
                wandb.log({
                    "Train run precision": train_prec,
                    "Train run recall": train_recall,
                    "Train run F1": train_f1,
                    "Train run G-Mean": train_gmean,
                    "Validation run precision": val_prec,
                    "Validation run recall": val_recall,
                    "Validation run F1": val_f1,
                    "Validation run G-Mean": val_gmean
                })
            mean_train_prec = np.mean(train_precs)
            mean_train_rec = np.mean(train_recs)
            mean_train_f1 = np.mean(train_f1s)
            mean_train_gmean = np.mean(train_gmeans)
            mean_val_prec = np.mean(val_precs)
            mean_val_rec = np.mean(val_recs)
            mean_val_f1 = np.mean(val_f1s)
            mean_val_gmean = np.mean(val_gmeans)

            # Log metrics
            wandb.summary["Train precision"] = mean_train_prec
            wandb.summary["Train recall"] = mean_train_rec
            wandb.summary["Train F1"] = mean_train_f1
            wandb.summary["Train G-Mean"] = mean_train_gmean
            wandb.summary["Validation precision"] = mean_val_prec
            wandb.summary["Validation recall"] = mean_val_rec
            wandb.summary["Validation F1"] = mean_val_f1
            wandb.summary["Validation G-Mean"] = mean_val_gmean