from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
import numpy as np
from time import time
from imblearn.metrics import geometric_mean_score

class RandomForestWrapper():
    # Hyperparameter combinations selected using validation G-Mean sweeps
    MEMSCAE_METRICS_PARAMETERS = {
        "n_estimators" : 2000,
        "max_depth" : 200,
        "max_features" : "sqrt",
        "max_samples" : 0.33,
    }
    SCAE_METRICS_PARAMETERS = {
        "n_estimators" : 1500,
        "max_depth" : 130,
        "max_features" : "log2",
        "max_samples" : 0.66,
    }
    BCAE_METRICS_PARAMETERS = {
        "n_estimators" : 250,
        "max_depth" : 100,
        "max_features" : "sqrt",
        "max_samples" : 0.33,
    }
    PCA_PARAMETERS = {
        "n_estimators" : 1200,
        "max_depth" : 3,
        "max_features" : "log2",
        "max_samples" : None,
    }
    
    SWEEP_PARAMETERS = {
        "pca": {
            "n_estimators": {
                "values": [800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200],
            },
            "max_depth": {
                "values": [3, 5, 7, 9, 11, 13],
            },
            "max_features": {
                "values": ["log2"],
            },
            "max_samples": {
                "values": [None],
            },
        },
        "memscae_metrics": {
            "n_estimators": {
                "values": [1750, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2500],
            },
            "max_depth": {
                "values": [175, 185, 190, 195, 200, 205, 210, 215, 250],
            },
            "max_features": {
                "values": ["sqrt"],
            },
            "max_samples": {
                "values": [0.33],
            },
        },
        "scae_metrics": {
            "n_estimators": {
                "values": [1350, 1400, 1450, 1500, 1550, 1600, 1650],
            },
            "max_depth": {
                "values": [110, 115, 120, 125, 130, 135, 140],
            },
            "max_features": {
                "values": ["log2"],
            },
            "max_samples": {
                "values": [0.55, 0.66, 0.77],
            },
        },
        "bcae_metrics": {
            "n_estimators": {
                "values": [100, 150, 200, 250, 300, 350, 400],
            },
            "max_depth": {
                "values": [85, 90, 95, 100, 105, 110, 115],
            },
            "max_features": {
                "values": ["sqrt"],
            },
            "max_samples": {
                "values": [0.22, 0.33, 0.44],
            },
        },
    }

    # Miscellaneous parameters
    random_state=42
    n_jobs = 8
    
    def __init__(self, mode):
        """Initializes the random forest wrapper.

        Args:
            mode (String): Indicates whether the model will receive PCA components or AE metrics.
        """
        self.mode = mode
    
    def set_default_model(self):
        """
        Create a random forest model using the default parameters.
        """
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
    
    def set_tuned_model(self):
        """
        Create a random forest model using the tuned parameters.
        """
        if self.mode == "pca":
            hparams = self.PCA_PARAMETERS
        elif self.mode == "memscae_metrics":
            hparams = self.MEMSCAE_METRICS_PARAMETERS
        elif self.mode == "scae_metrics":
            hparams = self.SCAE_METRICS_PARAMETERS
        elif self.mode == "bcae_metrics":
            hparams = self.BCAE_METRICS_PARAMETERS
        self.model =  RandomForestClassifier(
            n_estimators=hparams["n_estimators"],
            max_depth=hparams["max_depth"],
            max_features=hparams["max_features"],
            max_samples=hparams["max_samples"],
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
    
    def set_model_params(self, n_estimators, max_depth, max_features, max_samples):
        """Creates a random forest model using the given parameters.

        Args:
            n_estimators (int): The number of individual trees to use in the ensemble.
            max_depth (int): The maximum depth to restrict the trees to.
            max_features (float): The number of features to look at when choosing a split in the trees.
            max_samples (float): The percentage of samples to use for training each tree.
        """
        self.model =  RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=self.random_state,
            max_samples=max_samples,
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
        """Fit the RandomForest model on the given data.

        Args:
            train_X (ndarray): The training data.
            train_y (ndarray): The labels corresponding to the training data.
            val_X (ndarray): The validation data. Not used for this model.
            val_y (ndarray): The labels corresponding to the validation data. Not used for this model.
        """
        self.model.fit(train_X, train_y)
    
    def get_predictions(self, X):
        """Generate predictions for the given samples.

        Args:
            X (ndarray): The samples for which to generate anomaly predictions.

        Returns:
            ndarray: The predictions corresponding to the given samples.
        """
        return self.model.predict(X)
    
    def get_proba_predictions(self, X):
        """Generate probability predictions for the given samples

        Args:
            X (ndarray): The samples for which to generate anomaly probability predictions.

        Returns:
            ndarray: The probability predictions corresponding to the given samples.
        """
        return self.model.predict_proba(X)
    
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
                "n_estimators": {
                    "values": [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
                },
                "max_depth": {
                    "values": [5, 10, 25, 50, 75, 100, 125, 150, 175, 200],
                },
                "max_features": {
                    "values": ["sqrt", "log2"],
                },
                "max_samples": {
                    "values": [None, 0.33, 0.66],
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