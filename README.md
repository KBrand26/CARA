# CARA
The experiments in this repository consisted of training, testing and comparing various models for detecting anomalous radio sources. In this repository we make use of various convetional machine learning models as well as a few convolutional autoencoders for this purpose. 

# Data
## Download dataset
Prior to conducting any experiments, you will need to acquire the FRGADB dataset from:
```
https://zenodo.org/records/13773680
```
and download it into the **data/** directory.

## Prepare dataset
If you want to use the datamodules in this repository, you will need to execute **restructure_data.py** from the **data/** directory first. This will simply restructure the FRGADB dataset such that it is compatible with the existing datamodules.

This should be followed by executing **create_holdout_test.py**, which will generate the holdout test set. This script should be executed prior to running any experiments and should not be executed again until all experiments have been conducted.

# Project layout
  - **data/**: Contains all of the data and logic necessary for data loading, preprocessing and ingestion.
  - **callbacks/**: Contains the custom callbacks designed for these experiments.
  - **models/**: Contains all of the model architectures, hyperparameters and logic.
  - **src/**: Contains the experiment drivers which conduct the experiments.

For more information, refer to the README files in each of these directories.
