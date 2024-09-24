import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import correlate2d
from skimage.feature import match_template
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def load_data(dir_path, val=True, train=False):
    """This function is used to load the relevant outputs that were saved after AE training.

    Args:
        dir_path (String): The path to the folder containing the outputs from a specific run.
        val (bool, optional): Indicates whether only loading validation data. Defaults to True.
        train (bool, optional): Indicates whether training data should be loaded as well. Defaults to False.

    Returns:
        Tuple: A tuple containing the images, reconstructions, anomaly labels and true class labels.
    """
    if train:
        images = np.load(dir_path + 'train_images.npy')
        recons = np.load(dir_path + 'train_recons.npy')
        labels = np.load(dir_path + 'train_labels.npy')
        true_labels = np.load(dir_path + 'train_true_labels.npy')
    elif val:
        images = np.load(dir_path + 'val_images.npy')
        recons = np.load(dir_path + 'val_recons.npy')
        labels = np.load(dir_path + 'val_labels.npy')
        true_labels = np.load(dir_path + 'val_true_labels.npy')
    else:
        images = np.load(dir_path + 'test_images.npy')
        recons = np.load(dir_path + 'test_recons.npy')
        labels = np.load(dir_path + 'test_labels.npy')
        true_labels = np.load(dir_path + 'test_true_labels.npy')
    
    return images, recons, labels, true_labels

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

def normalized_cross_correlation(rec, og):
    """Calculates the normalized cross correlation between the original and reconstructed image.

    Args:
        rec (ndarray): The reconstructed image.
        og (ndarray): The original image.

    Returns:
        float: The maximum correlation after normalized cross-correlation.
    """
    ncc = match_template(og, rec, pad_input=True)    
    return ncc.max()

def cross_correlation(rec, og):
    """Calculates the cross-correlation between the original and reconstructed image. This unnormalized version should only be used with binary images.

    Args:
        rec (ndarray): The reconstructed image.
        og (ndarray): The original image.
    
    Returns:
        float: The maximum correlation value after cross-correlation.
    """
    cross_corr = correlate2d(rec, og, mode="same")
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].imshow(og, cmap='gray')
    ax[1].imshow(rec, cmap='gray')
    ax[2].imshow(cross_corr, cmap='gray')
    plt.show()
    
    return cross_corr.max()

def adapted_mse(rec, rec_thresh, og, og_thresh):
    """This function calculates an adapted mean squared error that focuses on the region of the images that contains the galaxy pixels.
    Thus this calculation is less biased to return smaller errors for smaller galaxies.

    Args:
        rec (ndarray): The reconstructed image.
        rec_thresh (ndarray): A thresholded version of the reconstructed that highlights the area containing the galaxy pixels.
        og (ndarray): The original image.
        og_thresh (ndarray): A thresholded version of the original image that highlights the area containing the galaxy pixels.

    Returns:
        float: The mean squared error of the galaxy regions.
    """
    # Determine the region to compare
    combined = np.logical_or(rec_thresh, og_thresh)
    og = og[combined]
    rec = rec[combined]
    
    # Calculate the error
    diff = og-rec
    squared = np.square(diff)
    return squared.mean()

def binary_mse(rec, og):
    """
    This function is similar to the adapted MSE, but is made specifically for binary, thresholded images.

    Args:
        rec (ndarray): The reconstructed image after thresholding.
        og (ndarray): The original image after thresholding.

    Returns:
        float: The error between the two given images.
    """
    # Determine the region that should be compared
    combined = np.logical_or(rec, og)
    rec = rec[combined]
    og = og[combined]
    
    # Calculate the error
    diff = og-rec 
    squared = np.square(diff)
    return squared.mean()

def calculate_adapted_bhattacharya(rec, rec_thresh, og, og_thresh):
    """This function calculates the bhattacharya distance between the original and reconstructed
    image, but only considers the regions that contain galaxy pixels. 

    Args:
        rec (ndarray): The reconstructed image.
        rec_thresh (ndarray): A thresholded version of the reconstructed image.
        og (ndarray): The original image.
        og_thresh (ndarray): A thresholded version of the reconstructed image.

    Returns:
        float: The Bhattacharya distance between the two images.
    """
    combined = np.logical_or(rec_thresh, og_thresh)
    og = og[combined]
    rec = rec[combined]

    # Convert image to probability distribution.
    rec = softmax(rec)
    og = softmax(og)
    comb = np.sqrt(og*rec)
    bhatta_coeff = comb.sum()
    return -1*np.log(bhatta_coeff)

def thresh_recon(image, mult=2.0):
    """Apply thresholding to the given reconstructed image to extract the shape of the reconstructed galaxy.

    Args:
        image (ndarray): The reconstructed image to threshold.
        mult (float): Indicates how many standard deviations above the mean to threshold. Defaults to 2.0.

    Returns:
        ndarray: The thresholded reconstruction.
    """
    mu = np.mean(image)
    std = np.std(image)
    threshed = image > (mu + mult*std)
    threshed = threshed.astype("float64")
    return threshed

def generate_dfs(dir_path, train=False):
    """Generates the dataframes containing the metric scores for the given autoencoder run.

    Args:
        dir_path (String): The path to the directory containing the outputs for the run to use.
        train (bool, optional): Indicates whether nearest neighbor metrics should be calculated using training data. Defaults to False.
    """
    if train:
        train_imgs, train_recs, train_lbls, train_tlbls = load_data(dir_path=dir_path, val=False, train=True)
        train_score_df = calculate_score_df(train_imgs, train_recs, train_lbls, train_tlbls)
        train_score_df.to_csv(dir_path + "train_scores.csv", index=False)
        del train_score_df, train_imgs, train_recs, train_lbls, train_tlbls

    val_imgs, val_recs, val_lbls, val_tlbls = load_data(dir_path=dir_path, val=True)
    val_score_df = calculate_score_df(val_imgs,  val_recs, val_lbls, val_tlbls)
    val_score_df.to_csv(dir_path + "val_scores.csv", index=False)
    
    test_imgs, test_recs, test_lbls, test_tlbls = load_data(dir_path=dir_path, val=False)
    test_score_df = calculate_score_df(test_imgs, test_recs, test_lbls, test_tlbls)
    test_score_df.to_csv(dir_path + "test_scores.csv", index=False)

def extract_pca(img):
    """Extracts the two PCA components that describes the shape of the galaxies. Similar to what was done for
    rotational standardisation.

    Args:
        img (ndarray): The image for which to extract the PCA components.

    Returns:
        Tuple: A tuple containing the principal components and explained variance.
    """
    # Extract the coordinates of the galaxy pixels
    coords = [[], []]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                coords[0].append(j)
                coords[1].append(i)

    coords = np.array(coords)

    # Center the image at the mean coordinate (necessary before PCA)
    mean = np.mean(coords,axis=1)[:,np.newaxis]
    coords = (coords - mean)
    
    princ_comps, var, _ = np.linalg.svd(coords,full_matrices=False)

    return princ_comps, var

def calculate_jaccard(original, reconstruction):
    """Calculates the Jaccard Index for a given pair of original and reconstructed
    images.

    Args:
        original (ndarray): The original image.
        reconstruction (ndarray): The corresponding reconstructed image.

    Returns:
        float: The calculated Jaccard index.
    """
    intersection = np.logical_and(original, reconstruction)
    union = np.logical_or(original, reconstruction)
    jaccard = intersection.sum()/union.sum()
    
    return jaccard

def calculate_score_df(imgs, recs, lbls, tlbls):
    """Calculates all of the metrics for a given set of images, latent encodings, reconstructions, anomaly labels and true labels.

    Args:
        imgs (ndarray): The array of images to use.
        recs (ndarray): The corresponding array of reconstructed images.
        lbls (ndarray): The anomaly labels corresponding to the images.
        tlbls (ndarray): The true class labels corresponding to the images.
    Returns:
        DataFrame: A dataframe containing all of the calculated metrics.
    """
    basic_mses = []
    nccs = []
    adapted_mses = []
    thresholded_mses = []
    skeleton_mses = []
    bhatts = []
    thresh_cos1s = []
    thresh_cos2s = []
    thresh_exp_var_maes = []
    thresh_jaccards = []
    skel_jaccards = []
    skel_cross_correlation = []
    for img, rec in tqdm(zip(imgs, recs)):
        # Generate thresholds and skeletons
        img_thresh = thresh_recon(img)
        pca_img_thresh, var_img_thresh = extract_pca(img_thresh[0])
        exp_var_og = ((var_img_thresh[0]**2)/2) / np.sum(np.square(var_img_thresh)/[2])
        rec_thresh = thresh_recon(rec)
        pca_rec_thresh, var_rec_thresh = extract_pca(rec_thresh[0])
        exp_var_rec = ((var_rec_thresh[0]**2)/2) / np.sum(np.square(var_rec_thresh)/[2])
        thresh_cos1 = -1*np.abs(cosine_similarity(pca_img_thresh[:, 0:1].reshape((1, -1)), pca_rec_thresh[:, 0:1].reshape((1, -1))))
        thresh_cos2 = -1*np.abs(cosine_similarity(pca_img_thresh[:, 1:].reshape((1, -1)), pca_rec_thresh[:, 1:].reshape((1, -1))))
        thresh_exp_var_mae = abs(exp_var_og - exp_var_rec)
        img_skel = skeletonize(img_thresh[0]).astype(float)
        rec_skel = skeletonize(rec_thresh[0]).astype(float)

        # Calculate the metrics and add them to the aggregate arrays. Some metrics are multiplied by -1 to ensure that they can all be minimized and not maximized.
        nccs.append(-1*normalized_cross_correlation(rec=rec[0], og=img[0]))
        basic_mses.append(basic_mse(rec=rec, og=img))
        adapted_mses.append(adapted_mse(rec=rec, rec_thresh=rec_thresh, og=img, og_thresh=img_thresh))
        bhatts.append(calculate_adapted_bhattacharya(rec=rec, rec_thresh=rec_thresh, og=img, og_thresh=img_thresh))
        thresholded_mses.append(binary_mse(rec_thresh, img_thresh))
        skeleton_mses.append(binary_mse(rec_skel, img_skel))
        thresh_jaccards.append(-1*calculate_jaccard(img_thresh, rec_thresh))
        skel_jaccards.append(-1*calculate_jaccard(img_skel, rec_skel))
        skel_cross_correlation.append(-1*normalized_cross_correlation(rec=rec_skel, og=img_skel))
        thresh_cos1s.append(thresh_cos1)
        thresh_cos2s.append(thresh_cos2)
        thresh_exp_var_maes.append(thresh_exp_var_mae)
    basic_mses = np.array(basic_mses)
    nccs = np.array(nccs)
    adapted_mses = np.array(adapted_mses)
    thresholded_mses = np.array(thresholded_mses)
    skeleton_mses = np.array(skeleton_mses)
    thresh_jaccards = np.array(thresh_jaccards)
    skel_jaccards = np.array(skel_jaccards)
    skel_cross_correlation = np.array(skel_cross_correlation)
    bhatts = np.array(bhatts)
    thresh_cos1s = np.array(thresh_cos1s)
    thresh_cos2s = np.array(thresh_cos2s)
    thresh_exp_var_maes = np.array(thresh_exp_var_maes)
    df = pd.DataFrame({
        "Basic MSE": basic_mses.reshape(-1),
        "Normalized Cross-Correlation": nccs.reshape(-1),
        "Adapted MSE": adapted_mses.reshape(-1),
        "Bhattacharya": bhatts.reshape(-1),
        "Thresholded MSE": thresholded_mses.reshape(-1),
        "Skeletonized MSE": skeleton_mses.reshape(-1),
        "Thresholded Jaccard Index": thresh_jaccards.reshape(-1),
        "Skeletonized Jaccard Index": skel_jaccards.reshape(-1),
        "Skeletonized Cross-Correlation": skel_cross_correlation.reshape(-1),
        "Thresholded PCA Cosine 1": thresh_cos1s.reshape(-1),
        "Thresholded PCA Cosine 2": thresh_cos2s.reshape(-1),
        "Thresholded PCA Explained Variance MAE": thresh_exp_var_maes.reshape(-1),
    })
    
    df["Label"] = lbls
    df["True Label"] = tlbls
    df["Combined MSE"] = (df["Adapted MSE"] + df["Thresholded MSE"] + df["Skeletonized MSE"])/3
    
    return df

if __name__ == "__main__":
    runs_to_use = [
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run1/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run2/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run3/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run4/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run5/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run6/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run7/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run8/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run9/",
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering_run10/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run1/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run2/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run3/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run4/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run5/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run6/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run7/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run8/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run9/",
        "saved_ae_data/SCAE_resampling_filtering_nn_run10/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run1/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run2/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run3/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run4/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run5/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run6/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run7/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run8/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run9/",
        "saved_ae_data/BCAE_resampling_filtering_nn_run10/",
    ]
    for i, run in enumerate(runs_to_use):
        print(f"Starting calculations for run {i}")
        generate_dfs(run, train=True)
    