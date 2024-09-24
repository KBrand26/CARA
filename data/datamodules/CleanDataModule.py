import lightning.pytorch as pl
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.insert(0,"data/datasets")
from BaseDataset import BaseDataset
from EvalDataset import EvalDataset
from skimage.transform import rotate
from astropy.stats import sigma_clip
from skimage.morphology import dilation, square

def find_connected_component(mask, row, col):
    """
    Makes use of morphological dilations to construct and extract the connected component
    at the given row and column in the mask.

    Args:
        mask (ndarray): The mask within which to find the connected component.
        row (int): The row of the starting pixel for the connected component.
        col (int): The column of the starting pixel for the connected component.

    Returns:
        tuple: A tuple containing the extracted connected component, as well as its size.
    """
    comp = np.zeros_like(mask)
    comp[row, col] = 1.0

    se = square(3)

    prev_comp = np.copy(comp)
    comp = np.multiply(dilation(comp, se), mask)

    while (comp - prev_comp).any():
        prev_comp = np.copy(comp)
        comp = np.multiply(dilation(comp, se), mask)

    return comp, np.sum(comp)

def remove_clipping_artefacts(mask, thresh=10):
    """Removes small artefacts from a given mask that are unlikely to belong to the galaxy.

    Args:
        mask (ndarray): A mask that was calculated during sigma clipping.
        thresh (int, optional): The threshold to use when determining whether a component is too small. 
            Defaults to 10.

    Returns:
        ndarray: Returns the mask after all small clipping artefacts have been removed.
    """
    new_mask = np.zeros_like(mask)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                comp, size = find_connected_component(mask, r, c)
                mask -= comp
                if size > thresh:
                    new_mask += comp
    return new_mask

class CleanDataModule(pl.LightningDataModule):
    """
    This datamodule encapsulates all of the logic for splitting samples and creating dataloaders
    to use during training/testing. This datamodule creates a "clean" training dataset that contains
    no anomalies. Its purpose is to train autoencoders.
    """
    def __init__(self, root_dir, standard_cls, anomaly_cls, batch_size, seed,
                 full_test=True, augment=False, filtering=False, resample=True):
        """This function is used to initialize the datamodule. It is more verbose than one would
        generally expect from a lightning datamodule. This is due to the logic that would generally
        be in the setup function being moved to the init, which was necessary for our use case.

        Args:
            root_dir (str): The root directory from which we will be extracting the data.
            standard_cls (List[str]): A list of classes to use as the standard samples.
            anomaly_cls (List[str]): A list of classes to use as the anomalies.
            batch_size (int): The number of samples to use in a batch.
            seed (int): A seed to use when initializing the various shufflers used to separate the datasets.
            full_test (bool, optional). Flag that indicates whether the test dataloader should load
              the full test set in a batch. Defaults to True.
            augment (bool, optional): Flag that indicates whether rotational augmentations should be
              used. Defaults to False.
            filtering (bool, optional): Flag that indicates whether noise filtering should be
              used. Defaults to False.
            resample (bool, optional): Flag that indicates whether resampling should be applied to the data.
                Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root_dir
        self.standard_cls = standard_cls
        self.anomaly_cls = anomaly_cls
        self.seed = seed
        self.batch_size = batch_size
        self.full_test = full_test
        self.augmentation = augment
        self.filter = filtering
        self.resamp = resample
        
        self.lbl_mapping = dict()
        
        # Normally, one would split the data in the setup function of a datamodule
        # However, in our case we had problems getting the pytorch lightning trainer to use the datamodule correctly.
        # Thus, the setup function is not called, which means that we have to setup the datasets here.
        self.all_train_standard_data = []
        self.all_test_standard_data = []
        self.all_train_standard_labels = []
        self.all_test_standard_labels = []
        self.all_train_standard_srcs = []
        self.all_test_standard_srcs = []
        for idx, tc in enumerate(self.standard_cls):
            self.lbl_mapping[tc] = idx
            # Load images and create labels for training
            tmp_train_X = list(np.load(f"{self.root}{tc}/train_X.npy"))
            tmp_train_src = list(np.load(f"{self.root}{tc}/train_src.npy"))
            tmp_train_y = [idx]*len(tmp_train_X)
            self.all_train_standard_data.extend(tmp_train_X)
            self.all_train_standard_labels.extend(tmp_train_y)
            self.all_train_standard_srcs.extend(tmp_train_src)
            
            tmp_test_X = list(np.load(f"{self.root}{tc}/test_X.npy"))
            tmp_test_src = list(np.load(f"{self.root}{tc}/test_src.npy"))
            tmp_test_y = [idx]*len(tmp_test_X)
            self.all_test_standard_data.extend(tmp_test_X)
            self.all_test_standard_labels.extend(tmp_test_y)
            self.all_test_standard_srcs.extend(tmp_test_src)
        self.all_train_standard_data = np.array(self.all_train_standard_data)
        self.all_train_standard_labels = np.array(self.all_train_standard_labels)
        self.all_test_standard_data = np.array(self.all_test_standard_data)
        self.all_test_standard_labels = np.array(self.all_test_standard_labels)
        self.all_train_standard_srcs = np.array(self.all_train_standard_srcs)
        self.all_test_standard_srcs = np.array(self.all_test_standard_srcs)
        
        # Load data from anomaly classes
        self.all_train_anomaly_data = []
        self.all_train_anomaly_labels = []
        self.all_test_anomaly_data = []
        self.all_test_anomaly_labels = []
        self.all_train_anomaly_srcs = []
        self.all_test_anomaly_srcs = []
        for tec in self.anomaly_cls:
            # Load class images and create labels
            tmp_train_X = list(np.load(f"{self.root}{tec}/ano_train_X.npy"))
            tmp_train_src = list(np.load(f"{self.root}{tec}/ano_train_src.npy"))
            tmp_train_y = [tec]*len(tmp_train_X)
            self.all_train_anomaly_data.extend(tmp_train_X)
            self.all_train_anomaly_labels.extend(tmp_train_y)
            self.all_train_anomaly_srcs.extend(tmp_train_src)
            
            tmp_test_X = list(np.load(f"{self.root}{tec}/ano_test_X.npy"))
            tmp_test_src = list(np.load(f"{self.root}{tec}/ano_test_src.npy"))
            tmp_test_y = [tec]*len(tmp_test_X)
            self.all_test_anomaly_data.extend(tmp_test_X)
            self.all_test_anomaly_labels.extend(tmp_test_y)
            self.all_test_anomaly_srcs.extend(tmp_test_src)
        self.all_train_anomaly_data = np.array(self.all_train_anomaly_data)
        self.all_train_anomaly_labels = np.array(self.all_train_anomaly_labels)
        self.all_train_anomaly_srcs = np.array(self.all_train_anomaly_srcs)
        self.all_test_anomaly_data = np.array(self.all_test_anomaly_data)
        self.all_test_anomaly_labels = np.array(self.all_test_anomaly_labels)
        self.all_test_anomaly_srcs = np.array(self.all_test_anomaly_srcs)
        self.anomaly_val_y = np.array([1]*len(self.all_train_anomaly_labels))

        # Create KFold generator for later use
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        self.fold_generator = skf.split(
            self.all_train_standard_data,
            self.all_train_standard_labels
        )

        # Call function to prepare training and validation sets. This is necessary, because these sets will change during experimentation.
        self.create_next_train_val_sets()
        
        # We can already prepare the test set here, because it will remain the same throughout all of the runs
        self.anomaly_test_y = np.array([1]*len(self.all_test_anomaly_labels))
        self.standard_test_y = np.array([0]*len(self.all_test_standard_labels))
        self.test_X = np.concatenate([self.all_test_standard_data, self.all_test_anomaly_data])
        self.test_true_y = np.concatenate([self.all_test_standard_labels, self.all_test_anomaly_labels])
        self.test_y = np.concatenate([self.standard_test_y, self.anomaly_test_y])
        self.test_srcs = np.concatenate([self.all_test_standard_srcs, self.all_test_anomaly_srcs])
        
        # Normalize test set
        self.test_X = self.normalize_images(self.test_X)
        # Filter background noise
        if self.filter:
            self.test_X = self.filter_images(self.test_X)
        # Create test dataset
        self.test = EvalDataset(self.test_X, self.test_y, self.test_true_y, self.test_srcs)
        
        print(f"Label mapping: {self.lbl_mapping}")
    
    def create_next_train_val_sets(self):
        """
        Iterates the KFolds generator and creates the new training and validation datasets.
        """
        # Split training data into train/val
        train_idxs, val_idxs = next(self.fold_generator)
        self.train_X = self.all_train_standard_data[train_idxs]
        self.train_true_y = self.all_train_standard_labels[train_idxs]
        self.train_srcs = self.all_train_standard_srcs[train_idxs]
        self.train_y = np.array([0]*len(self.train_true_y))
        if self.resamp:
            self.train_X, self.train_y, self.train_true_y, self.train_srcs = self.resample(self.train_X, self.train_y, self.train_true_y, self.train_srcs)

        self.std_val_X = self.all_train_standard_data[val_idxs]
        self.std_val_true_y = self.all_train_standard_labels[val_idxs]
        self.std_val_srcs = self.all_train_standard_srcs[val_idxs]
        self.std_val_y = np.array([0]*len(self.std_val_true_y))

        self.val_full_X = np.concatenate([self.std_val_X, self.all_train_anomaly_data])
        self.val_full_true_y = np.concatenate([self.std_val_true_y, self.all_train_anomaly_labels])
        self.val_full_srcs = np.concatenate([self.std_val_srcs, self.all_train_anomaly_srcs])
        self.val_full_y = np.concatenate([self.std_val_y, self.anomaly_val_y])

        # Rotational augmentation
        if self.augmentation:
            self.train_X, self.train_y, self.train_true_y, self.train_srcs = self.augment_images(self.train_X, self.train_y, self.train_true_y, self.train_srcs)
            
            # Shuffle datasets to improve training
            train_idxs = np.random.permutation(len(self.train_X))
            self.train_X = self.train_X[train_idxs]
            self.train_y = self.train_y[train_idxs]
            self.train_true_y = self.train_true_y[train_idxs]
            self.train_srcs = self.train_srcs[train_idxs]
        # Normalize
        self.train_X = self.normalize_images(self.train_X)
        self.std_val_X = self.normalize_images(self.std_val_X)
        self.val_full_X = self.normalize_images(self.val_full_X)
        # Filter background noise
        if self.filter:
            self.train_X = self.filter_images(self.train_X)
            self.std_val_X = self.filter_images(self.std_val_X)
            self.val_full_X = self.filter_images(self.val_full_X)
        self.train = BaseDataset(self.train_X, self.train_y, self.train_true_y)
        self.std_val = BaseDataset(self.std_val_X, self.std_val_y, self.std_val_true_y)
        self.full_val = EvalDataset(self.val_full_X, self.val_full_y, self.val_full_true_y, self.val_full_srcs)
    
    def prepare_data(self):
        """
        This function is required for this to be a valid datamodule, but we do not require it.
        """
        pass
    
    def setup(self, stage):
        """This function is used to set up the datasets.

        Args:
            stage (str): Indicates whether the model is fitting or testing
        """
        print(f"Setup is called with stage: {stage}")
    
    def resample(self, X, y, true_y, srcs):
        """Resample the given array to address class imbalance.

        Args:
            X (ndarray): The array containing the samples.
            y (ndarray): The array containing the normality labels.
            true_y (ndarray): The array containing the true class labels.
            srcs (ndarray): The array containing the source files of the samples.
            
        Returns:
            tuple: Returns a tuple containing the new X, y, true_y and src arrays after resampling.
        """
        # Determine minority and majority classes
        vals, counts = np.unique(true_y, return_counts=True)
        min_count_idx = np.argmin(counts)
        max_count_idx = np.argmax(counts)
        min_cls = vals[min_count_idx]
        maj_cls = vals[max_count_idx]
        total_size = counts[min_count_idx] + counts[max_count_idx]
        min_target = int(0.4*total_size)
        maj_target = int(0.6*total_size)
        
        # Oversample minority class
        if counts[min_count_idx] < min_target:
            min_indxs = np.where(true_y == min_cls)[0]
            size_increase = min_target-len(min_indxs)
            random_indxs = np.random.choice(min_indxs, size_increase, replace=False)
            min_indxs = np.append(min_indxs, random_indxs)
            min_X = X[min_indxs]
            min_y = y[min_indxs]
            min_true_y = true_y[min_indxs]
            min_srcs = srcs[min_indxs]
        else:
            min_indxs = np.where(true_y == min_cls)[0]
            min_X = X[min_indxs]
            min_y = y[min_indxs]
            min_true_y = true_y[min_indxs]
            min_srcs = srcs[min_indxs]
            
        # Undersample majority class
        if counts[max_count_idx] > maj_target:
            maj_indxs = np.where(true_y == maj_cls)[0]
            random_indxs = np.random.choice(maj_indxs, maj_target, replace=False)
            maj_X = X[random_indxs]
            maj_y = y[random_indxs]
            maj_true_y = true_y[random_indxs]
            maj_srcs = srcs[random_indxs]
        else:
            maj_indxs = np.where(true_y == maj_cls)[0]
            maj_X = X[maj_indxs]
            maj_y = y[maj_indxs]
            maj_true_y = true_y[maj_indxs]
            maj_srcs = srcs[maj_indxs]
            
        new_X = np.append(min_X, maj_X, axis = 0)
        new_y = np.append(min_y, maj_y)
        new_true_y = np.append(min_true_y, maj_true_y)
        new_srcs = np.append(min_srcs, maj_srcs)
        
        shuffle_idxs = np.random.permutation(len(new_X))
        new_X = new_X[shuffle_idxs]
        new_y = new_y[shuffle_idxs]
        new_true_y = new_true_y[shuffle_idxs]
        new_srcs = new_srcs[shuffle_idxs]
        
        return new_X, new_y, new_true_y, new_srcs

    def augment_images(self, X, y, true_y, srcs):
        """Performs rotational augmentation of a given dataset and its labels

        Args:
            X (ndarray): The images in the dataset.
            y (ndarray): The anomaly labels corresponding to the given images.
            true_y (ndarray): The true class labels corresponding to the given images.
            srcs (ndarray): The source files corresponding to the given images.

        Returns:
            tuple: Returns a tuple containing the augmented images and their corresponding
                anomaly and true class labels, as well as the name of their source files.
        """
        results = list(map(self.augment_image, X, y, true_y, srcs))
        new_Xs = []
        new_ys = []
        new_true_ys = []
        new_srcs = []
        for res in results:
            new_Xs.extend(res[0])
            new_ys.extend(res[1])
            new_true_ys.extend(res[2])
            new_srcs.extend(res[3])
        new_Xs = np.array(new_Xs)
        new_ys = np.array(new_ys)
        new_true_ys = np.array(new_true_ys)
        new_srcs = np.array(new_srcs)
        return new_Xs, new_ys, new_true_ys, new_srcs
    
    def augment_image(self, X, y, true_y, src):
        """Augments a given image and its labels, by rotating the image a number of times and
        duplicating the labels appropriately.

        Args:
            X (ndarray): The image to augment.
            y (int): The anomaly label corresponding to the image.
            true_y (int): The true class label corresponding to the given image.
            src (String): The source file corresponding to the given image.

        Returns:
           tuple: Returns a tuple containing the augmented images and the corresponding
                anomaly and true class labels.
        """
        augmented_X = [X]
        augmented_y = [y]
        augmented_true_y = [true_y]
        augmented_src = [src]
        for deg in range(45,360,45):
            augmented_X.append(rotate(X.copy(),deg))
            augmented_y.append(y)
            augmented_true_y.append(true_y)
            augmented_src.append(src)
        return augmented_X, augmented_y, augmented_true_y, augmented_src
    
    def filter_images(self, X):
        """Filters the background noise out of the given images.

        Args:
            X (ndarray): The array of images to filter.

        Returns:
            ndarray: The images after the background noise has been filtered out.
        """
        return np.array(list(map(self.filter_image, X)))
    
    def filter_image(self, img):
        """Makes use of sigma clipping and morphological operators to extract the galaxies from the given iamges.

        Args:
            img (ndarray): The image on which to apply filtering.

        Returns:
            ndarray: The resulting image after the background noise has been removed.
        """
        # Extract the pixels with extreme values
        mask = sigma_clip(img, maxiters=3).mask
        # Remove as many small clipping artefacts as possible
        mask = remove_clipping_artefacts(mask.astype("float32"))
        # Use the mask to extract only the galaxy pixels.
        filtered = mask*img
        
        return filtered
    
    def normalize_images(self, X):
        """Normalizes the given set of images

        Args:
            X (ndarray): The array of images to normalize.

        Returns:
            ndarray: The array of normalized images.
        """
        return np.array(list(map(self.normalize_image, X)))
        
    def normalize_image(self, img):
        """Normalize a given image

        Args:
            img (ndarray): The image to normalize

        Returns:
            ndarray: The normalized image
        """
        bot = np.min(img)
        top = np.max(img)
        norm = (img - bot)/(top - bot)
        return norm
    
    def train_dataloader(self):
        """Creates a dataloader for the training dataset

        Returns:
            DataLoader: A dataloader that iterates through the training dataset.
        """
        print("Ensure that create_next_train_val_sets has been called if this is not the first run.")
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        """Creates a dataloader for the validation dataset

        Returns:
            DataLoader: A dataloader that iterates through the validation dataset.
        """
        return DataLoader(self.std_val, batch_size=self.batch_size, num_workers=0, pin_memory=True)
    
    def ano_val_dataloader(self):
        """Creates a dataloader for the validation dataset with anomalous samples included.

        Returns:
            DataLoader: A dataloader that iterates through the full validation dataset.
        """
        return DataLoader(self.full_val, batch_size=self.batch_size, num_workers=0, pin_memory=True)
    
    def test_dataloader(self):
        """Creates a dataloader for the testing dataset

        Returns:
            DataLoader: A dataloader that iterates through the testing dataset.
        """
        if self.full_test:
            return DataLoader(self.test, batch_size=len(self.test), num_workers=0, pin_memory=True)
        else:
            return DataLoader(self.test, batch_size=self.batch_size, num_workers=0, pin_memory=True)
    
    
if __name__ == "__main__":
    mbcdm = CleanDataModule("data/FRGADB_Numpy/", [10, 20], [40, 50], 64, 42, augment=True, filtering=True)
    dl = mbcdm.train_dataloader()