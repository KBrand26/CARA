import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from imblearn.metrics import geometric_mean_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def calculate_supervised_metric_performance(val_df, test_df, metric_name):
    """Calculates the thresholds to use for the given metric using class labels and then evaluates the performance that this threshold can achieve
    for both the validation and the test set.

    Args:
        val_df (DataFrame): The dataframe containing the validation dataset metric values.
        test_df (DataFrame): The dataframe containing the testing dataset metric values.
        metric_name (String): The name of the metric for which to calculate the performance.

    Returns:
        DataFrame: A dataframe containing the performance metrics for the evaluated error metric.
    """
    scaler = MinMaxScaler()
    
    # Scale metric values to the [0, 1] range
    val_pred_y = val_df[metric_name].values
    val_pred_y = scaler.fit_transform(val_pred_y.reshape(-1, 1))
    val_y = val_df["Label"]
    test_pred_y = test_df[metric_name].values
    test_pred_y = scaler.transform(test_pred_y.reshape(-1, 1))
    test_y = test_df["Label"]

    # Calculate threshold and validation metrics
    thresh_min, thresh_max = val_pred_y.min(), val_pred_y.max()
    pos_thresholds = np.linspace(thresh_min, thresh_max, 100)
    pos_gmeans = []
    for thresh in pos_thresholds:
        val_pred_y_thresh = (val_pred_y > thresh).astype(int)
        pos_gmeans.append(geometric_mean_score(val_y, val_pred_y_thresh, average="binary"))
    arg_best_gmean = np.argmax(pos_gmeans)
    ano_thresh = pos_thresholds[arg_best_gmean]
    
    val_pred_y = (val_pred_y > ano_thresh).astype(int)
    val_gmean = geometric_mean_score(val_y, val_pred_y, average="binary")
    val_f1 = f1_score(val_y, val_pred_y, average="binary", zero_division=0.0)
    val_prec = precision_score(val_y, val_pred_y, average="binary", zero_division=0.0)
    val_rec = recall_score(val_y, val_pred_y, average="binary", zero_division=0.0)
    val_spec = recall_score(val_y, val_pred_y, average="binary", zero_division=0.0, pos_label=0)
    val_f2 = fbeta_score(val_y, val_pred_y, beta=2, average="binary", zero_division=0.0)
    
    # Calculate test metrics
    test_pred_y = (test_pred_y > ano_thresh).astype(int)
    test_gmean = geometric_mean_score(test_y, test_pred_y, average="binary")
    test_f1 = f1_score(test_y, test_pred_y, average="binary", zero_division=0.0)
    test_prec = precision_score(test_y, test_pred_y, average="binary", zero_division=0.0)
    test_rec = recall_score(test_y, test_pred_y, average="binary", zero_division=0.0)
    test_spec = recall_score(test_y, test_pred_y, average="binary", zero_division=0.0, pos_label=0)
    test_f2 = fbeta_score(test_y, test_pred_y, beta=2, average="binary", zero_division=0.0)
    
    new_row = pd.DataFrame({
        "Score": [metric_name + " Supervised"],
        "Val G-Mean": [val_gmean],
        "Val F1": [val_f1],
        "Val F2": [val_f2],
        "Val Spec": [val_spec],
        "Val Rec": [val_rec],
        "Val Prec": [val_prec],
        "Test G-Mean": [test_gmean],
        "Test F1": [test_f1],
        "Test F2": [test_f2],
        "Test Spec": [val_spec],
        "Test Rec": [test_rec],
        "Test Prec": [test_prec],
    })
    return new_row

def calculate_all_metric_performances(path, runs):
    """Evaluates the performance of all of the individual error metrics with respect to anomaly detection.

    Args:
        path (String): The base path to the error metrics directory for the given autoencoder architecture.
        runs (int): The number of runs conducted for the given architecture.
    """
    aggregate_eval = []
    first = True  
    for r in range(1, runs+1):
        print(f"Starting on run {r}....")
        val_df = pd.read_csv(path + f"_run{r}/" + "val_scores.csv")
        test_df = pd.read_csv(path + f"_run{r}/" "test_scores.csv")
        
        eval_df = pd.DataFrame(columns=["Score", "Val F1", "Val Prec", "Val Rec", "Test F1", "Test Prec", "Test Rec"])
        
        ignore_cols = ["Label", "True Label"]
        for score in val_df.columns:
            if score in ignore_cols:
                continue

            supervised_metric_row = calculate_supervised_metric_performance(val_df=val_df, test_df=test_df, metric_name=score)
            eval_df = pd.concat([eval_df, supervised_metric_row], ignore_index=True)
        eval_df = eval_df.sort_values(by="Test G-Mean", ascending=False).reset_index(drop=True)    
        eval_df.to_csv(path+f"_run{r}/metric_scores.csv", index=False)

        if first:
            vals = eval_df.values
            scores = vals[:, 0]
            aggregate_eval.append(vals[:, 1:])
            first = False
        else:
            vals = eval_df.values
            aggregate_eval.append(vals[:, 1:])
    aggregate_eval = np.stack(aggregate_eval, axis=-1).astype(np.float64)
    avg_eval = np.mean(aggregate_eval, axis=2)
    avg_eval_df = pd.DataFrame(data=avg_eval, columns=eval_df.columns[1:])
    avg_eval_df.insert(loc=0, column="Score", value=scores)
    os.makedirs(path, exist_ok=True)
    avg_eval_df.to_csv(path+"/avg_metric_scores.csv", index=False)
    
    std_eval = np.std(aggregate_eval, axis=2)
    std_eval_df = pd.DataFrame(data=std_eval, columns=eval_df.columns[1:])
    std_eval_df.insert(loc=0, column="Score", value=scores)
    std_eval_df.to_csv(path+"/std_metric_scores.csv", index=False)

if __name__ == "__main__":
    paths = [
        "saved_ae_data/memory_size500_shrink0.002_no_entropy_0.0002_resampling_filtering",
        "saved_ae_data/BCAE_resampling_filtering_nn",
        "saved_ae_data/SCAE_resampling_filtering_nn"
    ]
    
    for path in paths:
        calculate_all_metric_performances(path=path, runs=10)