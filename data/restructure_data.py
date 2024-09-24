from astropy.io import fits
import numpy as np
from os import makedirs
from os.path import exists
import glob

def format_frgadb_data():
    """
    This function is intended to restructure the publicly available version of the FRGADB such that it
    is compatible with the datamodules in this repository.
    """
    if not exists("data/FRGADB"):
        print("Please ensure that the FRGADB dataset has been downloaded into the data/ directory")
        return
    
    # Make the new directory structure
    makedirs("data/FRGADB_Numpy/10")
    makedirs("data/FRGADB_Numpy/20")
    makedirs("data/FRGADB_Numpy/40")
    makedirs("data/FRGADB_Numpy/50")
    
    # Prepare FRI
    paths = glob.glob("data/FRGADB/FRI/*.fits")
    fri_images = []
    fri_srcs = []
    for path in paths:
        # Open FITS cutout and extract the central 150x150 image
        img = fits.open(path)[0].data[75:225, 75:225]
        fri_images.append(img)
        fri_srcs.append(path)
    np.save("data/FRGADB_Numpy/10/X.npy", fri_images)
    np.save("data/FRGADB_Numpy/10/src.npy", fri_srcs)
    
    # Prepare FRII
    paths = glob.glob("data/FRGADB/FRII/*.fits")
    frii_images = []
    frii_srcs = []
    for path in paths:
        # Open FITS cutout and extract the central 150x150 image
        img = fits.open(path)[0].data[75:225, 75:225]
        frii_images.append(img)
        frii_srcs.append(path)
    np.save("data/FRGADB_Numpy/20/X.npy", frii_images)
    np.save("data/FRGADB_Numpy/20/src.npy", frii_srcs)
    
    # Prepare XRG
    paths = glob.glob("data/FRGADB/XRG/*.fits")
    xrg_images = []
    xrg_srcs = []
    for path in paths:
        # Open FITS cutout and extract the central 150x150 image
        img = fits.open(path)[0].data[75:225, 75:225]
        xrg_images.append(img)
        xrg_srcs.append(path)
    np.save("data/FRGADB_Numpy/40/X.npy", xrg_images)
    np.save("data/FRGADB_Numpy/40/src.npy", xrg_srcs)
    
    # Prepare RRG
    paths = glob.glob("data/FRGADB/RRG/*.fits")
    rrg_images = []
    rrg_srcs = []
    for path in paths:
        # Open FITS cutout and extract the central 150x150 image
        img = fits.open(path)[0].data[75:225, 75:225]
        rrg_images.append(img)
        rrg_srcs.append(path)
    np.save("data/FRGADB_Numpy/50/X.npy", rrg_images)
    np.save("data/FRGADB_Numpy/50/src.npy", rrg_srcs)
    
if __name__ == "__main__":
    format_frgadb_data()