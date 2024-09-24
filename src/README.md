# Guide
## Script explanations:
 - **ae_driver.py**: The main driver script for training autoencoders. At the end of the file there are examples of using this script to train the various autoencoders used for the experiments.
 - **calculate_metrics.py**: This script is used after training autoencoders to calculate all of the metrics that compare the inputs and the outputs.
 - **metric_evaluation.py**: This script is used to evaluate the individual performance of each error metric with respect to anomaly detection, using only basic thresholding strategies.
 - **model_constructor.py**: This scipt is mainly used by the driver scripts to construct the models that will be used. It is not meant for isolated use.
 - **ml_driver.py**: The main driver script for training the machine learning models that will be used for anomaly detection. Can be used to conduct hyperparameter sweeps as well as final testing runs. There are examples at the end of the file indicating how to use the script for PCA or metric data.

## Suggested usage:
 1. Use **ae_driver.py** to train the chosen autoencoder architecture, ensure that the save flag is true to save the reconstructions. Otherwise, you will need to reload the saved model checkpoint at a later point to generate these reconstructions.
 2. Run **calculate_metrics.py** to calculate the metrics for your autoencoder outputs.
 3. Make use of **metric_evaluation.py** to evaluate the anomaly detection performance of the individual performance metrics.
 4. Make use of **ml_driver.py** to train the anomaly detection models on both the generated metrics and PCA output. You can also repeat the hyperparameter sweeps for yourself, but you will then need to update the tuned parameters for each model by updating their respective wrappers in the **model/ml/** directory.