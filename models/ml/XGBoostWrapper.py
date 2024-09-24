import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
import numpy as np
from time import time
from imblearn.metrics import geometric_mean_score

class XGBoostWrapper():
    """
    This class serves as a wrapper for XGBoost models
    """
    # Hyperparameter combinations selected using validation G-Mean sweeps
    MEMSCAE_METRICS_PARAMETERS = {
        "n_estimators" : 1250,
        "max_depth" : 50,
        "learning_rate" : 0.4,
        "gamma" : 2,
        "subsample" : 1,
        "colsample_bytree" : 0.33,
        "reg_lambda" : 4,
    }
    SCAE_METRICS_PARAMETERS = {
        "n_estimators" : 1300,
        "max_depth" : 40,
        "learning_rate" : 0.03,
        "gamma" : 7,
        "subsample" : 0.66,
        "colsample_bytree" : 0.95,
        "reg_lambda" : 7,
    }
    BCAE_METRICS_PARAMETERS = {
        "n_estimators" : 1750,
        "max_depth" : 75,
        "learning_rate" : 0.05,
        "gamma" : 4,
        "subsample" : 0.66,
        "colsample_bytree" : 0.66,
        "reg_lambda" : 6,
    }
    PCA_PARAMETERS = {
        "n_estimators" : 300,
        "max_depth" : 5,
        "learning_rate" : 0.01,
        "gamma" : 11,
        "subsample" : 0.77,
        "colsample_bytree" : 1,
        "reg_lambda" : 2,
    }
    
    SWEEP_PARAMETERS = {
        "pca": {
            "n_estimators" : {
                "values":[100, 150, 200, 250, 300, 350, 400],
            },
            "max_depth" : {
                "values":[3, 5, 7, 9, 11],
            },
            "learning_rate" : {
                "values":[0.001, 0.005, 0.01, 0.015, 0.02],
            },
            "gamma" : {
                "values":[7, 8, 9, 10, 11],
            },
            "subsample" : {
                "values":[0.55, 0.66, 0.77],
            },
            "colsample_bytree" : {
                "values":[0.9, 0.95, 1],
            },
            "reg_lambda" : {
                "values":[1, 2, 3],
            },
        },
        "memscae_metrics": {
            "n_estimators" : {
                "values":[1100, 1150, 1200, 1250, 1300, 1350, 1400],
            },
            "max_depth" : {
                "values":[35, 40, 45, 50, 55, 60, 65],
            },
            "learning_rate" : {
                "values":[0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45],
            },
            "gamma" : {
                "values":[1, 2, 3],
            },
            "subsample" : {
                "values":[0.9, 0.95, 1],
            },
            "colsample_bytree" : {
                "values":[0.22, 0.33, 0.44],
            },
            "reg_lambda" : {
                "values":[3, 4, 5],
            },
        },
        "scae_metrics": {
            "n_estimators" : {
                "values":[1100, 1150, 1200, 1250, 1300, 1350, 1400],
            },
            "max_depth" : {
                "values":[35, 40, 45, 50, 55, 60, 65],
            },
            "learning_rate" : {
                "values":[0.001, 0.005, 0.008, 0.009, 0.01, 0.02, 0.03, 0.05, 0.1],
            },
            "gamma" : {
                "values":[7, 8, 9, 10],
            },
            "subsample" : {
                "values":[0.55, 0.66, 0.77],
            },
            "colsample_bytree" : {
                "values":[0.9, 0.95, 1],
            },
            "reg_lambda" : {
                "values":[5, 6, 7],
            },
        },
        "bcae_metrics": {
            "n_estimators" : {
                "values":[1600, 1650, 1700, 1750, 1800, 1850, 1900],
            },
            "max_depth" : {
                "values":[60, 65, 70, 75, 80, 85, 90],
            },
            "learning_rate" : {
                "values":[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            },
            "gamma" : {
                "values":[3, 4, 5],
            },
            "subsample" : {
                "values":[0.55, 0.66, 0.77],
            },
            "colsample_bytree" : {
                "values":[0.55, 0.66, 0.77],
            },
            "reg_lambda" : {
                "values":[5, 6, 7],
            },
        }
    }

    device="cuda"
    random_state=42
    early_stopping_rounds=10
    objective="binary:logistic"
    
    def __init__(self, mode):
        """Initializes the wrapper

        Args:
            mode (String): Indicates what the input to the models will be.
        """
        self.mode = mode
    
    def set_default_model(self):
        """
        This function sets up the XGBoost model with default settings.
        """
        self.model = xgb.XGBClassifier(device=self.device, random_state=self.random_state, \
            early_stopping_rounds=self.early_stopping_rounds)
    
    def set_tuned_model(self):
        """
        Sets up a XGBoost model using the finetuned parameters from a hyperparameter sweep.
        """
        if self.mode == "pca":
            hparams = self.PCA_PARAMETERS
        elif self.mode == "memscae_metrics":
            hparams = self.MEMSCAE_METRICS_PARAMETERS
        elif self.mode == "scae_metrics":
            hparams = self.SCAE_METRICS_PARAMETERS
        elif self.mode == "bcae_metrics":
            hparams = self.BCAE_METRICS_PARAMETERS
        self.model =  xgb.XGBClassifier(
            n_estimators=hparams["n_estimators"],
            max_depth=hparams["max_depth"],
            learning_rate=hparams["learning_rate"],
            objective=self.objective,
            gamma=hparams["gamma"],
            subsample=hparams["subsample"],
            colsample_bytree=hparams["colsample_bytree"],
            reg_lambda=hparams["reg_lambda"],
            device=self.device,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds
        )
    
    def set_model_params(self, n_estimators, max_depth, learning_rate, gamma, subsample, \
        colsample_bytree, reg_lambda):
        """Sets up a XGBoost model using the specific parameters that were provided.

        Args:
            n_estimators (int): The number of boosting rounds to use.
            max_depth (int): The maximum depth that individual trees are allowed to reach.
            learning_rate (float): The learning rate to use during boosting.
            gamma (float): The minimum amount of loss reduction required to make another partition in a tree.
            subsample (float): The ratio of samples to use for each training run.
            reg_lambda (float): L2 regularization weight.
        """
        self.model =  xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=self.objective,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            device=self.device,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds
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
        """Fit the XGBoost model on the given data.

        Args:
            train_X (ndarray): The training data.
            train_y (ndarray): The labels corresponding to the training data.
            val_X (ndarray): The validation data.
            val_y (ndarray): The labels corresponding to the validation data.
        """
        self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)])
    
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
    
    def get_sweep_config(self, name, first_sweep=False):
        """Generates the configuration file for the hyperparameter sweep.

        Args:
            name (String): The name to use for the sweep.
            first_sweep (bool, optional): Indicates whether this is the first sweep or the finetuned sweep. Defaults to True.
        """
        if first_sweep:
            parameters = {
                "n_estimators" : {
                    "values":[100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
                },
                "max_depth" : {
                    "values":[5, 10, 25, 50, 75, 100],
                },
                "learning_rate" : {
                    "values":[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                },
                "gamma" : {
                    "values":[0, 2, 4, 6, 8],
                },
                "subsample" : {
                    "values":[0.33, 0.66, 1],
                },
                "colsample_bytree" : {
                    "values":[0.33, 0.66, 1],
                },
                "reg_lambda" : {
                    "values":[0, 2, 4, 6, 8],
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
                train_X, train_y, val_X, val_y, _, _, _, _ = self.data_func()     
                
                # Set model parameters
                self.set_model_params(
                    **config
                )

                # Train model
                self.fit_model(train_X, train_y, val_X, val_y)
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