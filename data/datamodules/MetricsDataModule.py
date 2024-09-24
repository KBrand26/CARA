import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from time import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MetricsDataModule():
    """
    This data class encapsulates all of the data preparation logic for standard anomaly detection models on the metrics generated from the results of an autoencoder.
    """
    # Selected metrics
    METRICS = [
        "Basic MSE",
        "Skeletonized MSE",
        "Thresholded PCA Cosine 1",
        "Thresholded PCA Cosine 2",
        "Thresholded PCA Explained Variance MAE"
    ]
    
    def __init__(self, root_dir, repeats, seed=42, use_train_normal=True, one_class=False):
        """The initialisation function for this data class.

        Args:
            root_dir (str): The root directory from which we will be extracting the data.
            repeats (int). The number of 5-folds repetitions that are required.
            seed (int, optional): A seed to use when initializing the various shufflers used to separate the datasets. Defaults to 42.
            use_train_normal (bool, optional):  Flag that indicates whether the normal samples from CAE training should be used as well. Defaults to True.
            one_class (bool, optional): Flag that indicates whether a single anomaly class should be used for training.
        """
        super().__init__()
        self.root = root_dir
        self.seed = seed
        self.repeats = repeats
        self.oc = one_class
        self.use_train_normal = use_train_normal
        
        # Load the data from all the requested classes
        self.load_data()
        
        # Create kfolds instance
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=self.repeats, random_state=self.seed)

        self.fold_generator = skf.split(
            self.full_train_X, self.full_train_true_y
        )
        
        # Create first training and validation subsets
        self.next_train_val_split()
    
    def load_data(self):
        """
        This function is used to load the saved metrics data.
        """
        only_train_scores_df = pd.read_csv(self.root + 'train_scores.csv')
        train_scores_df = pd.read_csv(self.root + 'val_scores.csv')
        test_scores_df = pd.read_csv(self.root + 'test_scores.csv')

        if self.use_train_normal:
            self.only_train_X = only_train_scores_df[self.METRICS].to_numpy()
            self.only_train_y = only_train_scores_df["Label"].to_numpy()
            self.only_train_true_y = only_train_scores_df["True Label"].to_numpy()
        
        self.full_train_X = train_scores_df[self.METRICS].to_numpy()
        self.full_train_y = train_scores_df["Label"].to_numpy()
        self.full_train_true_y = train_scores_df["True Label"].to_numpy()
        
        # Not called test_X to preserve original test data
        self.og_test_X = test_scores_df[self.METRICS].to_numpy()
        self.test_y = test_scores_df["Label"].to_numpy()
        self.test_true_y = test_scores_df["True Label"].to_numpy()

    def next_train_val_split(self):
        """
        This function generates the new subset of training and validation data and applies all preprocessing
        required to the new subsets.
        """
        # Split training data into train/val
        try:
            train_idxs, val_idxs = next(self.fold_generator)
        except:
            # If this datamodule is called more times than there are folds, fail gracefully
            print("No more folds remaining...")
            return
        self.train_X, self.val_X = self.full_train_X[train_idxs], self.full_train_X[val_idxs]
        self.train_y, self.val_y = self.full_train_y[train_idxs], self.full_train_y[val_idxs]
        self.train_true_y, self.val_true_y = self.full_train_true_y[train_idxs], self.full_train_true_y[val_idxs]
        
        if self.use_train_normal:
            # Add the normal samples that can only be used for training
            self.train_X = np.concatenate([self.train_X, self.only_train_X])
            self.train_y = np.concatenate([self.train_y, self.only_train_y])
            self.train_true_y = np.concatenate([self.train_true_y, self.only_train_true_y])
        
        # Remove ring-like galaxies if doing testing with a single class
        if self.oc:
            not_ring_idxs = self.train_true_y != 50
            self.train_X = self.train_X[not_ring_idxs]
            self.train_y = self.train_y[not_ring_idxs]
            self.train_srcs = self.train_srcs[not_ring_idxs]
            self.train_true_y = self.train_true_y[not_ring_idxs]
        
        # Normalize the metrics
        self.normalize_metrics()

        # Shuffle to ensure that symmetry is broken
        self.train_X, self.train_y, self.train_true_y = self.shuffle_data(self.train_X, self.train_y, self.train_true_y)

    def shuffle_data(self, X, y, true_y):
        """This functions shuffles the given data.

        Args:
            X (ndarray): An array containing the metrics data.
            y (ndarray): An array containing the anomaly labels corresponding to the metrics.
            true_y (ndarray): An array containing the true class labels corresponding to the metrics.

        Returns:
            tuple: A tuple containing the shuffled arrays.
        """
        shuffle_idxs = np.random.permutation(len(X))
        X = X[shuffle_idxs]
        y = y[shuffle_idxs]
        true_y = true_y[shuffle_idxs]
        
        return X, y, true_y
    
    def normalize_metrics(self):
        """
        Normalizes the data subsets.
        """
        scaler = MinMaxScaler()
        # Only fit on the training data to prevent leakage into the unseen samples.
        self.train_X = scaler.fit_transform(self.train_X)
        self.val_X = scaler.transform(self.val_X)
        self.test_X = scaler.transform(self.og_test_X)
    
    def get_datasets(self):
        """
        Fetches the data subsets.

        Returns:
            tuple: A tuple containing the metrics and corresponding labels for the training, validation and testing subsets.
        """
        return self.train_X, self.train_y, self.val_X, self.val_y, self.val_true_y, self.test_X, self.test_y, self.test_true_y