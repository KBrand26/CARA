import numpy as np
from astropy.stats import sigma_clip
from skimage.morphology import dilation, square
from sklearn.model_selection import RepeatedStratifiedKFold
from skimage.transform import rotate
from sklearn.decomposition import PCA
from time import time
from sklearn.preprocessing import MinMaxScaler

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

class PCADataModule():
    """
    This data class encapsulates all of the data preparation logic for standard anomaly detection models. It makes use
    of PCA to reduce the dimensionality of the dataset appropriately.
    """
    def __init__(self, root_dir, standard_cls, anomaly_cls, repeats, n_components=0.7, seed=42,
                 augment=False, filtering=False, dim_norm=True):
        """The initialisation function for this data class.

        Args:
            root_dir (str): The root directory from which we will be extracting the data.
            standard_cls (List[str]): A list of classes to use as the standard samples.
            anomaly_cls (List[str]): A list of classes to use as the anomalies.
            repeats (int). The number of repetitions of 5-folds that are required.
            n_components (float or int): The number of components to keep during PCA. A float value can be used to indicate percentage
                of explained variance to keep or an int can be used for the exact number of components. Defaults to 0.7.
            seed (int): A seed to use when initializing the various shufflers used to separate the datasets. Defaults to 42.
            augment (bool, optional): Flag that indicates whether rotational augmentations should be
              used. Defaults to False.
            filtering (bool, optional): Flag that indicates whether noise filtering should be
              used. Defaults to False.
            dim_norm (bool, optional): Flag that indicates whether dimensionality reduction output needs to be normalized.
        """
        super().__init__()
        self.root = root_dir
        self.standard_cls = standard_cls
        self.anomaly_cls = anomaly_cls
        self.seed = seed
        self.augmentation = augment
        self.filter = filtering
        self.repeats = repeats
        self.n_components = n_components
        self.dim_norm = dim_norm
        
        # Load the data from all of the requested classes
        self.load_data()
        
        # Apply normalization and filtering (only needs to happen once, won't be affected by splits, because it is per image)
        self.full_X = self.normalize_images(self.full_X)
        self.test_X_imgs = self.normalize_images(self.test_X_imgs)
        if self.filter:
            self.full_X = self.filter_images(self.full_X)
            self.test_X_imgs = self.filter_images(self.test_X_imgs)
        
        # Create kfolds instance
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=self.repeats, random_state=self.seed)

        self.fold_generator = skf.split(
            self.full_X, self.full_true_y
        )
        
        # Create first training and validation subsets
        self.next_train_val_split()
    
    def shuffle_data(self, X, y, true_y, srcs):
        """This functions shuffles the given data.

        Args:
            X (ndarray): An array containing the PCA data.
            y (ndarray): An array containing the anomaly labels corresponding to the PCA data.
            true_y (ndarray): An array containing the true class labels corresponding to the PCA data.
            srcs (ndarray): An array containing the name of the original filenames corresponding to the PCA data.

        Returns:
            tuple: A tuple containing the shuffled arrays.
        """
        shuffle_idxs = np.random.permutation(len(X))
        X = X[shuffle_idxs]
        y = y[shuffle_idxs]
        true_y = true_y[shuffle_idxs]
        srcs = srcs[shuffle_idxs]
        
        return X, y, true_y, srcs
    
    def load_data(self):
        """
        This function is used to load and prepare the image data for PCA.
        """
        self.lbl_mapping = dict()
        all_train_standard_data = []
        all_test_standard_data = []
        all_train_standard_labels = []
        all_test_standard_labels = []
        all_train_standard_srcs = []
        all_test_standard_srcs = []
        for idx, tc in enumerate(self.standard_cls):
            self.lbl_mapping[tc] = idx
            # Load images and create labels for training
            tmp_train_X = list(np.load(f"{self.root}{tc}/train_X.npy"))
            tmp_train_src = list(np.load(f"{self.root}{tc}/train_src.npy"))
            tmp_train_y = [idx]*len(tmp_train_X)
            all_train_standard_data.extend(tmp_train_X)
            all_train_standard_labels.extend(tmp_train_y)
            all_train_standard_srcs.extend(tmp_train_src)

            tmp_test_X = list(np.load(f"{self.root}{tc}/test_X.npy"))
            tmp_test_src = list(np.load(f"{self.root}{tc}/test_src.npy"))
            tmp_test_y = [idx]*len(tmp_test_X)
            all_test_standard_data.extend(tmp_test_X)
            all_test_standard_labels.extend(tmp_test_y)
            all_test_standard_srcs.extend(tmp_test_src)
        all_train_standard_data = np.array(all_train_standard_data)
        all_train_standard_labels = np.array(all_train_standard_labels)
        all_test_standard_data = np.array(all_test_standard_data)
        all_test_standard_labels = np.array(all_test_standard_labels)
        all_train_standard_srcs = np.array(all_train_standard_srcs)
        all_test_standard_srcs = np.array(all_test_standard_srcs)
        standard_train_y = np.array([0]*len(all_train_standard_labels))
        standard_test_y = np.array([0]*len(all_test_standard_labels))
        
        # Load data from anomaly classes
        all_train_anomaly_data = []
        all_train_anomaly_labels = []
        all_test_anomaly_data = []
        all_test_anomaly_labels = []
        all_train_anomaly_srcs = []
        all_test_anomaly_srcs = []
        for tec in self.anomaly_cls:
            # Load class images and create labels
            tmp_train_X = list(np.load(f"{self.root}{tec}/ano_train_X.npy"))
            tmp_train_src = list(np.load(f"{self.root}{tec}/ano_train_src.npy"))
            tmp_train_y = [tec]*len(tmp_train_X)
            all_train_anomaly_data.extend(tmp_train_X)
            all_train_anomaly_labels.extend(tmp_train_y)
            all_train_anomaly_srcs.extend(tmp_train_src)
            
            tmp_test_X = list(np.load(f"{self.root}{tec}/ano_test_X.npy"))
            tmp_test_src = list(np.load(f"{self.root}{tec}/ano_test_src.npy"))
            tmp_test_y = [tec]*len(tmp_test_X)
            all_test_anomaly_data.extend(tmp_test_X)
            all_test_anomaly_labels.extend(tmp_test_y)
            all_test_anomaly_srcs.extend(tmp_test_src)
        all_train_anomaly_data = np.array(all_train_anomaly_data)
        all_train_anomaly_labels = np.array(all_train_anomaly_labels)
        all_train_anomaly_srcs = np.array(all_train_anomaly_srcs)
        all_test_anomaly_data = np.array(all_test_anomaly_data)
        all_test_anomaly_labels = np.array(all_test_anomaly_labels)
        all_test_anomaly_srcs = np.array(all_test_anomaly_srcs)
        anomaly_train_y = np.array([1]*len(all_train_anomaly_labels))
        anomaly_test_y = np.array([1]*len(all_test_anomaly_labels))

        self.full_X = np.concatenate([all_train_standard_data, all_train_anomaly_data])
        self.full_true_y = np.concatenate([all_train_standard_labels, all_train_anomaly_labels])
        self.full_srcs = np.concatenate([all_train_standard_srcs, all_train_anomaly_srcs])
        self.full_y = np.concatenate([standard_train_y, anomaly_train_y])
        
        self.test_X_imgs = np.concatenate([all_test_standard_data, all_test_anomaly_data])
        self.test_true_y = np.concatenate([all_test_standard_labels, all_test_anomaly_labels])
        self.test_y = np.concatenate([standard_test_y, anomaly_test_y])
        self.test_srcs = np.concatenate([all_test_standard_srcs, all_test_anomaly_srcs])
        
    def apply_pca(self, n_components=0.7):
        """
        This function applies PCA to the various subsets of images.

        Args:
            n_components (float, optional): Indicates the percentage of explained variance to keep
            after PCA. Defaults to 0.7.
        """
        # Flatten images for PCA
        flattened_train_X = self.flatten_images(self.train_X)
        flattened_val_X = self.flatten_images(self.val_X)
        flattened_test_X = self.flatten_images(self.test_X_imgs)
        
        # Train the PCA instance
        self.pca = PCA(n_components=n_components)
        self.pca.fit(flattened_train_X)

        # Apply PCA
        self.train_X = self.pca.transform(flattened_train_X)
        self.val_X = self.pca.transform(flattened_val_X)
        self.test_X = self.pca.transform(flattened_test_X)
    
    def flatten_images(self, images):
        """Flattens the given array of images

        Args:
            images (ndarray): An array containing the images that need to be flattened for PCA.

        Returns:
            ndarray : The array containing the flattened images.
        """
        return np.array(list(map(self.flatten_image, images)))
        
    def flatten_image(self, image):
        """Flattens a given image.

        Args:
            image (ndaray): Contains the image that needs to be flattened.

        Returns:
            ndarray: The flattened version of the given image.
        """
        return image.flatten()
    
    def next_train_val_split(self):
        """
        This function generates the new subset of training and validation data and applies all preprocessing
        required to the new subsets.
        """
        # Split training data into train/val
        try:
            train_idxs, val_idxs = next(self.fold_generator)
        except:
            print("No more folds remaining...")
            return
        self.train_X, self.val_X = self.full_X[train_idxs], self.full_X[val_idxs]
        self.train_y, self.val_y = self.full_y[train_idxs], self.full_y[val_idxs]
        self.train_true_y, self.val_true_y = self.full_true_y[train_idxs], self.full_true_y[val_idxs]
        self.train_srcs, self.val_srcs = self.full_srcs[train_idxs], self.full_srcs[val_idxs]
        
        # Augment training dataset to address rotations
        if self.augmentation:
            self.train_X, self.train_y, self.train_true_y, self.train_srcs = self.augment_images(self.train_X, self.train_y, self.train_true_y, self.train_srcs)

            # Shuffle datasets to improve training
            train_idxs = np.random.permutation(len(self.train_X))
            self.train_X = self.train_X[train_idxs]
            self.train_y = self.train_y[train_idxs]
            self.train_true_y = self.train_true_y[train_idxs]
            self.train_srcs = self.train_srcs[train_idxs]
        
        # Apply dimensionality reduction technique to the images
        self.apply_pca(n_components=self.n_components)
        
        # Normalize the output of dimensionality reduction
        if self.dim_norm:
            self.normalize_components()

        # Shuffle one last time
        self.train_X, self.train_y, self.train_true_y, self.train_srcs = self.shuffle_data(self.train_X, self.train_y, self.train_true_y, self.train_srcs)

    def normalize_components(self):
        """
        Normalizes the PCA components.
        """
        scaler = MinMaxScaler()
        self.train_X = scaler.fit_transform(self.train_X)
        self.val_X = scaler.transform(self.val_X)
        self.test_X = scaler.transform(self.test_X)
    
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
        """Makes use of sigma clipping and morphological operators to extract the galaxies from the given images.

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
    
    def get_datasets(self):
        return self.train_X, self.train_y, self.val_X, self.val_y, self.val_true_y, self.test_X, self.test_y, self.test_true_y
    
if __name__ == "__main__":
    mbcdm = PCADataModule("data/FRGADB_Numpy/", [10, 20], [40, 50], repeats=1, n_components=0.7, seed=42, augment=True, filtering=True, dim_norm=True)