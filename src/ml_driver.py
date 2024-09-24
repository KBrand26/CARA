import wandb
import numpy as np
import sys
sys.path.insert(0,"data/datamodules/")
sys.path.insert(0, "models/ml/")
from LocalOutlierFactorWrapper import LocalOutlierFactorWrapper
from RandomForestWrapper import RandomForestWrapper
from IsolationForestWrapper import IsolationForestWrapper
from XGBoostWrapper import XGBoostWrapper
from PCADataModule import PCADataModule
from MetricsDataModule import MetricsDataModule
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, fbeta_score
from imblearn.metrics import geometric_mean_score
from time import time
import pandas as pd
from matplotlib import pyplot as plt
import os

def per_class_accuracy(y_true, y_pred, y_class, class_lbl):
    """Calculates the per class accuracy for given labels.

    Args:
        y_true (ndarray): The true anomaly labels.
        y_pred (ndarray): The predicted anomaly labels.
        y_class (ndarray): The true class labels.
        class_lbl (ndarray): The class for which to calculate the accuracy.

    Returns:
        float: The accuracy for the given class.
    """
    class_idxs = y_class == class_lbl
    y_true = y_true[class_idxs]
    y_pred = y_pred[class_idxs]
    cls_acc = (y_true == y_pred).sum()/len(y_true)
    return cls_acc

def eval_model(train_y, train_y_pred, val_y, val_y_pred, test_y, test_y_pred):
    """Evaluate a model given the training, validation and testing labels.

    Args:
        train_y (ndarray): The anomaly labels for the training data.
        train_y_pred (ndarray): The anomaly label predictions for the training data. 
        val_y (ndarray): The anomaly labels for the validation data.
        val_y_pred (ndarray): The anomaly label predictions for the validation data.
        test_y (ndarray): The anomaly labels for the testing data.
        test_y_pred (ndarray): The anomaly label predictions for the testing data.

    Returns:
        tuple: A tuple containing the precision, recall, f1s, f2s and geometric means for all three splits.
    """
    cm = confusion_matrix(train_y, train_y_pred, normalize='all')
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    train_prec = tp/(tp+fp+1e-6)
    train_rec = tp/(tp+fn+1e-6)
    train_spec = tn/(tn+fp+1e-6)
    train_f1 = (2*train_prec*train_rec)/(train_prec+train_rec+1e-6)
    train_f2 = fbeta_score(y_true=train_y, y_pred=train_y_pred, beta=2, average="binary", zero_division=0.0)
    train_g_mean = geometric_mean_score(y_true=train_y, y_pred=train_y_pred, average="binary")

    cm = confusion_matrix(val_y, val_y_pred, normalize='all')
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    val_prec = tp/(tp+fp+1e-6)
    val_rec = tp/(tp+fn+1e-6)
    val_spec = tn/(tn+fp+1e-6)
    val_f1 = (2*val_prec*val_rec)/(val_prec+val_rec+1e-6)
    val_f2 = fbeta_score(y_true=val_y, y_pred=val_y_pred, beta=2, average="binary", zero_division=0.0)
    val_g_mean = geometric_mean_score(y_true=val_y, y_pred=val_y_pred, average="binary")

    cm = confusion_matrix(test_y, test_y_pred, normalize='all')
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    test_prec = tp/(tp+fp+1e-6)
    test_rec = tp/(tp+fn+1e-6)
    test_spec = tn/(tn+fp+1e-6)
    test_f1 = (2*test_prec*test_rec)/(test_prec+test_rec+1e-6)
    test_f2 = fbeta_score(y_true=test_y, y_pred=test_y_pred, beta=2, average="binary", zero_division=0.0)
    test_g_mean = geometric_mean_score(y_true=test_y, y_pred=test_y_pred, average="binary")
    
    return train_prec, train_rec, train_spec, train_f1, train_f2, train_g_mean, val_prec, val_rec, val_spec, val_f1, val_f2, \
        val_g_mean, test_prec, test_rec, test_spec, test_f1, test_f2, test_g_mean

class ExperimentRunner():
    MODEL_LABEL = {
        "rf": "random_forest_",
        "if": "isolation_forest_",
        "xgb": "xgboost_",
        "lof": "local_outlier_factor_",
    }
    SMOTE_LABEL = {
        True: "SMOTE_",
        False: ""
    }
    AUGMENT_LABEL = {
        True: "augment_",
        False: ""
    }
    FILTER_LABEL = {
        True: "filter_",
        False: ""
    }
    RESAMPLE_LABEL = {
        True: "resampling_",
        False: ""
    }
    NORM_LABEL = {
        True: "dimnorm_",
        False: ""
    }
    DEFAULT_LABEL = {
        True: "default_",
        False: ""
    }
    FIRST_SWEEP_LABEL = {
        True: "",
        False: "finetune"
    }
    
    def __init__(self, runs, path, repeats, mode, run_sub_name="", model="rf", sweep=False, first_sweep=False, smote=False, default=False, \
        augment=False, filtering=False, n_components=0.7, dim_norm=True, quali_analysis=False, train=False):
        """Initializes an instance of an experiment runner.

        Args:
            runs (int): Indicates how many runs to execute during experimentation.
            path (String): The path to the root directory containing the files to use when initializing the data module.
            repeats (int): Indicates the number of repeats needed during kfolds.
            mode (String): Indicates what mode to run in. Options include: pca, memscae_metrics,
                scae_metrics and bcae_metrics.
            run_sub_name (String): Any additional information to add to the run name. Defaults to "".
            model (String, optional): Indicates what model to use. Options include: rf, xgb, if, lof.
                Defaults to rf.
            sweep (bool, optional): Flag that indicates whether to conduct a hyperparameter sweep or not.
                Defaults to False.
            first_sweep (bool, optional): Flag that indicates whether this is the first sweep or the finetuned sweep.
                Defaults to False.
            smote (bool, optional): Flag that indicates whether SMOTE should be applied to the data.
                Defaults to False.
            default (bool, optional): Flag that indicates whether to use the default model parameters.
                Defaults to False.
            augment (bool, optional): A flag that indicates whether to augment the images in the dataset.
                Defaults to False.
            filtering (bool, optional): A flag that indicates whether to filter the images in the dataset to remove noise.
                Defaults to False.
            n_components (float, optional): Indicates the percentage of components to keep when using PCA.
                Defaults to 0.7.
            dim_norm (float, optional): Indicates whether outputs from dimensionality reduction should be normalized.
                Defaults to True.
            train (bool, optional): A flag that indicates whether to include the training data when using the
                error metrics dataset. Defaults to False.
        """
        self.path = path
        self.repeats = repeats
        self.runs = runs
        self.mode = mode
        self.run_sub_name = run_sub_name
        self.model_type = model
        self.sweep = sweep
        self.first_sweep = first_sweep
        self.smote = smote
        self.default = default
        self.augment = augment
        self.filtering = filtering
        self.n_components = n_components
        self.train = train
        self.dim_norm = dim_norm
        
        # Login to wandb
        wandb.login()

    def setup_pca_dm(self, n_components=0.7):
        """Sets up the PCA datamodule

        Args:
            n_components (float, optional): The percentage of the explained variance to preserve after PCA.
                Defaults to 0.7.
        """
        self.anomaly_classes = [40, 50]
        self.normal_classes = [10, 20]
        self.data = PCADataModule(self.path, self.normal_classes, self.anomaly_classes, self.repeats,
            n_components=n_components, seed=42, augment=self.augment, filtering=self.filtering, dim_norm=self.dim_norm)
        if n_components < 1:
            self.n_components = self.data.pca.n_components_
            print(f"Keeping {self.n_components} components when using {n_components*100:.2f}% of the variance.")
            del self.data
    
    def setup_metrics_dm(self):
        """
        Creates the data module for a run on the error metrics data
        """
        train_normal = self.model_type in ["if", "lof"]
        
        self.data = MetricsDataModule(self.path, repeats=self.repeats, seed=42, use_train_normal=train_normal, one_class=False)
    
    def print_run_start_info(self):
        """
        Simple utility function to announce the start of a run
        """
        print("=============================================================================================")
        print(f"Starting run {self.current_run}.")
        print("=============================================================================================")
    
    def conduct_model_training(self):
        """
        Conduct training of the current model. Also evaluates model performance.
        """
        # Train model
        self.model.fit_model(self.train_X, self.train_y, self.val_X, self.val_y)
        
        # Generate predictions for the data subsets
        train_y_pred = self.model.get_predictions(self.train_X)
        val_y_pred = self.model.get_predictions(self.val_X)
        test_y_pred = self.model.get_predictions(self.test_X)

        # Evaluate model outputs
        train_prec, train_rec, train_spec, train_f1, train_f2, train_g_mean, val_prec, val_rec, val_spec, val_f1, val_f2, val_g_mean, \
            test_prec, test_rec, test_spec, test_f1, test_f2, test_g_mean = eval_model(self.train_y, train_y_pred, self.val_y, \
                val_y_pred, self.test_y, test_y_pred)

        val_true_accs = []
        test_true_accs = []
        cls_lbls = np.unique(self.test_true_y)
        for cls_lbl in cls_lbls:
            val_true_acc = per_class_accuracy(y_true=self.val_y, y_pred=val_y_pred, y_class=self.val_true_y, class_lbl=cls_lbl)
            val_true_accs.append(val_true_acc)
            test_true_acc = per_class_accuracy(y_true=self.test_y, y_pred=test_y_pred, y_class=self.test_true_y, class_lbl=cls_lbl)
            test_true_accs.append(test_true_acc)
            wandb.summary[f"Validation class {cls_lbl} accuracy"] = val_true_acc
            wandb.summary[f"Test class {cls_lbl} accuracy"] = test_true_acc
        balanced_val_true_acc = np.mean(val_true_accs)
        balanced_test_true_acc = np.mean(test_true_accs)
        
        val_ano_accs = []
        test_ano_accs = []
        cls_lbls = [0, 1]
        for cls_lbl in cls_lbls:
            val_ano_acc = per_class_accuracy(y_true=self.val_y, y_pred=val_y_pred, y_class=self.val_y, class_lbl=cls_lbl)
            val_ano_accs.append(val_ano_acc)
            test_ano_acc = per_class_accuracy(y_true=self.test_y, y_pred=test_y_pred, y_class=self.test_y, class_lbl=cls_lbl)
            test_ano_accs.append(test_ano_acc)
            wandb.summary[f"Validation class {cls_lbl} accuracy"] = val_ano_acc
            wandb.summary[f"Test class {cls_lbl} accuracy"] = test_ano_acc
        balanced_val_ano_acc = np.mean(val_ano_accs)
        balanced_test_ano_acc = np.mean(test_ano_accs)

        run_results_row = pd.DataFrame({
            "Run": [self.run_name],
            "F1": [test_f1],
            "F2": [test_f2],
            "GMean": [test_g_mean],
            "Precision": [test_prec],
            "Recall": [test_rec],
            "Specificity": [test_spec],
            "Per-True-Class Accuracy": [balanced_test_true_acc],
            "Per-Anomaly-Class Accuracy": [balanced_test_ano_acc]
        })
        self.experiment_metric_df = pd.concat([self.experiment_metric_df, run_results_row], ignore_index=True)

        # Log metrics
        wandb.summary["Balanced true-class validation accuracy"] = balanced_val_true_acc
        wandb.summary["Balanced true-class test accuracy"] = balanced_test_true_acc
        wandb.summary["Balanced anomaly-class validation accuracy"] = balanced_val_ano_acc
        wandb.summary["Balanced anomaly-class test accuracy"] = balanced_test_ano_acc
        wandb.summary["Train precision"] = train_prec
        wandb.summary["Train recall"] = train_rec
        wandb.summary["Train specificity"] = train_spec
        wandb.summary["Train F1"] = train_f1
        wandb.summary["Train F2"] = train_f2
        wandb.summary["Train G-Mean"] = train_g_mean
        wandb.summary["Validation precision"] = val_prec
        wandb.summary["Validation recall"] = val_rec
        wandb.summary["Validation specificity"] = val_spec
        wandb.summary["Validation F1"] = val_f1
        wandb.summary["Validation F2"] = val_f2
        wandb.summary["Validation G-Mean"] = val_g_mean
        wandb.summary["Test precision"] = test_prec
        wandb.summary["Test recall"] = test_rec
        wandb.summary["Test specificity"] = test_spec
        wandb.summary["Test F1"] = test_f1
        wandb.summary["Test F2"] = test_f2
        wandb.summary["Test G-Mean"] = test_g_mean
        
    def execute_model_training(self):
        """
        This function is used to conduct training with appropriate parameters for the given scenario.
        """
        self.train_X, self.train_y, self.val_X, self.val_y, self.val_true_y, \
            self.test_X, self.test_y, self.test_true_y = self.get_data()
        
        self.conduct_model_training()
    
    def create_wrapper(self):
        """
        This function creates the model wrapper for the requested model type.
        """
        if self.model_type == "rf":
            self.model = RandomForestWrapper(self.mode)
        elif self.model_type == "xgb":
            self.model = XGBoostWrapper(self.mode)
        elif self.model_type == "if":
            self.model = IsolationForestWrapper(self.mode)
        elif self.model_type == "lof":
            self.model = LocalOutlierFactorWrapper(self.mode)
    
    def get_data(self):
        """This function is used to fetch the various data subsets

        Returns:
            Tuple: A tuple containing the train images and labels, validation images and labels
                and testing images and labels.
        """
        # Get data
        train_X, train_y, val_X, val_y, val_true_y, test_X, test_y, test_true_y = self.data.get_datasets()

        # Use SMOTE to address class imbalance
        if self.smote:
            smote = SMOTE()
            train_X, train_y = smote.fit_resample(train_X, train_y)

        # Indicate to the datamodule that it can prepare the next training/validation split in preparation for the next run
        self.data.next_train_val_split()
        return train_X, train_y, val_X, val_y, val_true_y, test_X, test_y, test_true_y
    
    def execute_sweep(self):
        """
        This function contains the logic for the conduction of a hyperparameter sweep
        """
        print("Hyperparameter sweep starting...")
        # Prepare data class
        if self.mode == "pca":
            self.setup_pca_dm(n_components=self.n_components)
        elif self.mode in ["memscae_metrics", "scae_metrics", "bcae_metrics"]:
            self.setup_metrics_dm()

        # Set the function in the wrapper for acquiring a new data split.
        self.model.set_sweep_data_function(self.get_data)

        # Conduct the sweep
        self.sweep_name = self.group_name + "sweep"
        self.model.conduct_wandb_sweep(self.sweep_name, self.runs, self.first_sweep)
        
        # Reset states
        self.model.cleanup_sweep()
        del self.data
        print("Hyperparameter sweep concluded")
    
    def execute_experiment(self):
        """
        This function serves as the main driver for the experiment.
        """
        # Create model wrapper
        self.create_wrapper()
        
        # Set group name
        model_lbl = self.MODEL_LABEL[self.model_type]
        augment = self.AUGMENT_LABEL[self.augment]
        smote = self.SMOTE_LABEL[self.smote]
        filter = self.FILTER_LABEL[self.filtering]
        default = self.DEFAULT_LABEL[self.default]
        ncomps = f"{self.n_components}_" if self.mode == "pca" else ""
        train = f"with_train_" if self.mode in ["memscae_metrics", "scae_metrics", "bcae_metrics"] \
            and self.train else ""
        dn = self.NORM_LABEL[self.dim_norm]
        first_sweep = self.FIRST_SWEEP_LABEL[self.first_sweep]
        self.group_name = f"{self.mode}_{train}{ncomps}{model_lbl}{augment}{smote}{filter}{dn}{default}{self.run_sub_name}_{first_sweep}_"
        
        # If using PCA, first conduct one full run of PCA to determine the number of components to use
        if self.mode == "pca":
            self.setup_pca_dm(n_components=self.n_components)

        if self.sweep:
            # Conduct a hyperparameter sweep
            self.execute_sweep()
        else:
            # Create dataframe for recording experiment stats
            self.experiment_metric_df = pd.DataFrame(columns=[
                "Run", "F1", "Precision", "Recall", "Per-True-Class Accuracy", "Per-Anomaly-Class Accuracy"
            ])
            
            if self.mode == "pca":
                self.setup_pca_dm(n_components=self.n_components)
            elif self.mode in ["memscae_metrics", "scae_metrics", "bcae_metrics"]:
                self.setup_metrics_dm()
            # Conduct full experiments
            for r in range(1, self.runs+1):
                start = time()
                self.execute_run(r)

                if r != self.runs:
                    del self.model.model
                    print(f"Completed run {r}")
                    start2 = time()
                    wandb.finish()
                    end2 = time()
                    print(f"Wandb finish took {end2-start2} seconds.")
                    end = time()
                    print(f"Run took {end-start} seconds.")
                else:
                    self.experiment_metric_df.to_csv(f"saved_ae_data/{self.group_name}.csv", index=False)
                    print("Experiment concluded")
                    wandb.finish()

    def execute_run(self, run):
        """Driver function for conducting a single experiment run.

        Args:
            run (int): The number of the run currently being executed.
        """
        # Set run details
        self.current_run = run
        self.run_name = f"{self.group_name}run{self.current_run}"
        
        # Create model instance
        if self.default:
            self.model.set_default_model()
        else:
            self.model.set_tuned_model()

        # Conduct run
        self.print_run_start_info()
        start = time()
        wandb.init(
            project="CARAShallow",
            group=self.group_name,
            name=self.run_name,
            reinit=True,
        )
        end = time()
        print(f"Wandb init took {end - start} second")
        self.execute_model_training()
        
if __name__ == "__main__":
    # These flags can be used to control the type of experiments being conducted
    # Sweep controls whether a hyperparameter sweep will be conducted
    sweep = False
    # First sweep indicates whether this is the first coarse sweep or the second finer sweep
    first_sweep = False
    # Established PCA Percentage indicates whether it is necessary to determine the percentage of PCA components to use
    establish_pca_perc = False
    # The number of runs
    ae_runs = 1
    
    # Indicate which datasets to use
    datasets = [
        "pca",
        "memscae_metrics",
        "scae_metrics",
        "bcae_metrics"
    ]
    # Add the paths to metric datasets if necessary
    metrics_paths = [
        "",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run",
        "saved_ae_data/SCAE_resampling_filtering_nn_run",
        "saved_ae_data/BCAE_resampling_filtering_nn_run"
    ]
    # Add additional information to the run names
    run_sub_names = [
        "",
        "",
        "",
        ""
    ]
    
    for data, metrics_path, run_sub_name in zip(datasets, metrics_paths, run_sub_names):
        if data == "pca":
            # Random Forest on PCA components
            if not sweep:
                # Tuned
                runner = ExperimentRunner(runs=10, path="data/FRGADB_Numpy/", repeats=2, mode="pca", run_sub_name=run_sub_name, \
                    model="rf", sweep=False, smote=True, default=False, augment=False, filtering=True, n_components=0.5, \
                        dim_norm=True)
                runner.execute_experiment()
            else:
                if establish_pca_perc:
                    # 0.5 components without augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="rf", sweep=True, first_sweep=first_sweep, smote=True, augment=False, filtering=True, n_components=0.5, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.7 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="rf", sweep=True, first_sweep=first_sweep, smote=True, augment=True, filtering=True, n_components=0.7, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.9 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="rf", sweep=True, first_sweep=first_sweep, smote=True, augment=True, filtering=True, n_components=0.9, \
                            dim_norm=True)
                    runner.execute_experiment()
                else:
                    # Inspect 60 hyperparameter combinations (already established that 0.5 PCA without augmentation tends
                    # to perform the best)
                    runner = ExperimentRunner(runs=60, path="data/FRGADB_Numpy/", repeats=120, mode="pca", run_sub_name=run_sub_name, \
                        model="rf", sweep=True, first_sweep=first_sweep, smote=True, augment=False, filtering=True, n_components=0.5, \
                            dim_norm=True)
                    runner.execute_experiment()
                    

            # XGB on PCA components
            if not sweep:
                # Tuned
                runner = ExperimentRunner(runs=10, path="data/FRGADB_Numpy/", repeats=2, mode="pca", run_sub_name=run_sub_name, \
                    model="xgb", sweep=False, smote=True, default=False, augment=True, filtering=True, n_components=0.7, \
                        dim_norm=True)
                runner.execute_experiment()
            else:
                if establish_pca_perc:
                    # 0.5 components without augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="xgb", sweep=True, first_sweep=first_sweep, smote=True, augment=False, filtering=True, n_components=0.5, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.7 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="xgb", sweep=True, first_sweep=first_sweep, smote=True, augment=True, filtering=True, n_components=0.7, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.9 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="xgb", sweep=True, first_sweep=first_sweep, smote=True, augment=True, filtering=True, n_components=0.9, \
                            dim_norm=True)
                    runner.execute_experiment()
                else:
                    # Inspect 60 hyperparameter combinations (already established that 0.7 PCA with augmentation tends
                    # to perform the best)
                    runner = ExperimentRunner(runs=60, path="data/FRGADB_Numpy/", repeats=120, mode="pca", run_sub_name=run_sub_name, \
                        model="xgb", sweep=True, first_sweep=first_sweep, smote=True, augment=True, filtering=True, n_components=0.7, \
                            dim_norm=True)
                    runner.execute_experiment()
            
            # Isolation Forest on PCA components
            if not sweep:
                # Default
                runner = ExperimentRunner(runs=10, path="data/FRGADB_Numpy/", repeats=2, mode="pca", run_sub_name=run_sub_name, \
                    model="if", sweep=False, smote=False, default=True, augment=True, filtering=True, n_components=0.7, \
                        dim_norm=True)
                runner.execute_experiment()
                
                # Tuned
                runner = ExperimentRunner(runs=10, path="data/FRGADB_Numpy/", repeats=2, mode="pca", run_sub_name=run_sub_name, \
                    model="if", sweep=False, smote=False, default=False, augment=True, filtering=True, n_components=0.7, \
                        dim_norm=True)
                runner.execute_experiment()
            else:
                if establish_pca_perc:
                    # 0.5 components without augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="if", sweep=True, first_sweep=first_sweep, smote=False, augment=False, filtering=True, n_components=0.5, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.7 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="if", sweep=True, first_sweep=first_sweep, smote=False, augment=True, filtering=True, n_components=0.7, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.9 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="if", sweep=True, first_sweep=first_sweep, smote=False, augment=True, filtering=True, n_components=0.9, \
                            dim_norm=True)
                    runner.execute_experiment()
                else:
                    # Inspect 60 hyperparameter combinations (already established that 0.7 PCA with augmentation tends
                    # to perform the best)
                    runner = ExperimentRunner(runs=60, path="data/FRGADB_Numpy/", repeats=120, mode="pca", run_sub_name=run_sub_name, \
                        model="if", sweep=True, first_sweep=first_sweep, smote=False, augment=True, filtering=True, n_components=0.7, \
                            dim_norm=True)
                    runner.execute_experiment()
                
            # Local Outlier Factor on PCA components
            if not sweep:
                # Default
                runner = ExperimentRunner(runs=10, path="data/FRGADB_Numpy/", repeats=2, mode="pca", run_sub_name=run_sub_name, \
                    model="lof", sweep=False, smote=False, default=True, augment=True, filtering=True, n_components=0.7, \
                        dim_norm=True)
                runner.execute_experiment()
                
                # Tuned
                runner = ExperimentRunner(runs=10, path="data/FRGADB_Numpy/", repeats=2, mode="pca", run_sub_name=run_sub_name, \
                    model="lof", sweep=False, smote=False, default=False, augment=False, filtering=True, n_components=0.5, \
                        dim_norm=True)
                runner.execute_experiment()
            else:
                if establish_pca_perc:
                    # 0.5 components without augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="lof", sweep=True, first_sweep=first_sweep, smote=False, augment=False, filtering=True, n_components=0.5, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.7 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="lof", sweep=True, first_sweep=first_sweep, smote=False, augment=True, filtering=True, n_components=0.7, \
                            dim_norm=True)
                    runner.execute_experiment()
                    # 0.9 components with augmentation
                    runner = ExperimentRunner(runs=20, path="data/FRGADB_Numpy/", repeats=40, mode="pca", run_sub_name=run_sub_name, \
                        model="lof", sweep=True, first_sweep=first_sweep, smote=False, augment=True, filtering=True, n_components=0.9, \
                            dim_norm=True)
                    runner.execute_experiment()
                else:
                    # Inspect 60 hyperparameter combinations (already established that 0.7 PCA with augmentation tends
                    # to perform the best)
                    runner = ExperimentRunner(runs=60, path="data/FRGADB_Numpy/", repeats=120, mode="pca", run_sub_name=run_sub_name, \
                        model="lof", sweep=True, first_sweep=first_sweep, smote=False, augment=True, filtering=True, n_components=0.7, \
                            dim_norm=True)
                    runner.execute_experiment()
        elif data in ["memscae_metrics", "scae_metrics", "bcae_metrics"]:
            # Random Forest on metrics
            if not sweep:
                for r in range(1, ae_runs+1):
                    metrics_run_path = metrics_path + f"{r}/"
                    run_full_sub_name = run_sub_name +f"{r}"
                    
                    # Tuned
                    runner = ExperimentRunner(runs=10, path=metrics_run_path, repeats=2, mode=data, \
                        run_sub_name=run_full_sub_name, model="rf", sweep=False, smote=True, default=False, dim_norm=True)
                    runner.execute_experiment()
            else:
                runner = ExperimentRunner(runs=60, path=metrics_path, repeats=120, mode=data, \
                    run_sub_name=run_sub_name, model="rf", sweep=True, first_sweep=first_sweep, smote=True, dim_norm=True)
                runner.execute_experiment()

            # XGBoost on metrics
            if not sweep:
                for r in range(1, ae_runs+1):
                    metrics_run_path = metrics_path + f"{r}/"
                    run_full_sub_name = run_sub_name +f"{r}"
                    
                    # Tuned
                    runner = ExperimentRunner(runs=10, path=metrics_run_path, repeats=2, mode=data, \
                        run_sub_name=run_full_sub_name, model="xgb", sweep=False, smote=True, default=False, dim_norm=True)
                    runner.execute_experiment()
            else:
                runner = ExperimentRunner(runs=60, path=metrics_path, repeats=120, mode=data, \
                    run_sub_name=run_sub_name, model="xgb", sweep=True, first_sweep=first_sweep, smote=True, dim_norm=True)
                runner.execute_experiment()

            # Isolation forest on metrics
            if not sweep:
                for r in range(1, ae_runs+1):
                    metrics_run_path = metrics_path + f"{r}/"
                    run_full_sub_name = run_sub_name +f"{r}"
                    # Default
                    runner = ExperimentRunner(runs=10, path=metrics_run_path, repeats=2, mode=data, \
                        run_sub_name=run_full_sub_name, model="if", sweep=False, smote=False, default=True, dim_norm=True)
                    runner.execute_experiment()
                    
                    # Tuned
                    runner = ExperimentRunner(runs=10, path=metrics_run_path, repeats=2, mode=data, \
                        run_sub_name=run_full_sub_name, model="if", sweep=False, smote=False, default=False, dim_norm=True)
                    runner.execute_experiment()
            else:
                runner = ExperimentRunner(runs=60, path=metrics_path, repeats=120, mode=data, \
                    run_sub_name=run_sub_name, model="if", sweep=True, first_sweep=first_sweep, smote=False, dim_norm=True)
                runner.execute_experiment()
            
            # # Local outlier factor on metrics
            if not sweep:
                for r in range(1, ae_runs+1):
                    metrics_run_path = metrics_path + f"{r}/"
                    run_full_sub_name = run_sub_name +f"{r}"
                    # Default
                    runner = ExperimentRunner(runs=10, path=metrics_run_path, repeats=2, mode=data, \
                        run_sub_name=run_full_sub_name, model="lof", sweep=False, smote=False, default=True, dim_norm=True)
                    runner.execute_experiment()
                    
                    # Tuned
                    runner = ExperimentRunner(runs=10, path=metrics_run_path, repeats=2, mode=data, \
                        run_sub_name=run_full_sub_name, model="lof", sweep=False, smote=False, default=False, dim_norm=True)
                    runner.execute_experiment()
            else:
                runner = ExperimentRunner(runs=60, path=metrics_path, repeats=120, mode=data, \
                    run_sub_name=run_sub_name, model="lof", sweep=True, first_sweep=first_sweep, smote=False, dim_norm=True)
                runner.execute_experiment()