import numpy as np
from sklearn.model_selection import train_test_split

def create_normal_splits(dir_path):
    """Splits the samples from a given normal class into training and testing splits

    Args:
        dir_path (str): The path to the root directory for the normal class.
    """
    X = np.load(dir_path + "X.npy")
    src = np.load(dir_path + "src.npy")
    
    train_X, test_X, train_src, test_src = train_test_split(
        X,
        src,
        test_size=0.1,
        shuffle=True,
        random_state=42,
    )

    np.save(dir_path+"train_X.npy", train_X)
    np.save(dir_path+"test_X.npy", test_X)
    np.save(dir_path+"train_src.npy", train_src)
    np.save(dir_path+"test_src.npy", test_src)

def create_ano_splits(dir_path):
    """Splits the samples from a given anomalous class into training and testing splits

    Args:
        dir_path (str): The path to the root directory for the anomalous class.
    """
    X = np.load(dir_path + "X.npy")
    src = np.load(dir_path + "src.npy")

    ano_train_X, ano_test_X, ano_train_src, ano_test_src = train_test_split(
        X,
        src,
        test_size=0.5,
        shuffle=True,
        random_state=42,
    )

    np.save(dir_path+"ano_train_X.npy", ano_train_X)
    np.save(dir_path+"ano_test_X.npy", ano_test_X)
    np.save(dir_path+"ano_train_src.npy", ano_train_src)
    np.save(dir_path+"ano_test_src.npy", ano_test_src)
    
def validate_normal_splits(dir_path):
    """Validates the splits created for each normal class.

    Args:
        dir_path (str): The path to the root directory for the class.
    """
    X = np.load(dir_path + "X.npy")
    src = np.load(dir_path + "src.npy")
    train_X = np.load(dir_path + "train_X.npy")
    test_X = np.load(dir_path + "test_X.npy")
    train_src = np.load(dir_path + "train_src.npy")
    test_src = np.load(dir_path + "test_src.npy")
    
    total = X.shape[0]
    train_perc = int((train_X.shape[0]/total)*100)
    test_perc = int((test_X.shape[0]/total)*100)
    
    src_total = src.shape[0]
    src_train_perc = int((train_src.shape[0]/src_total)*100)
    src_test_perc = int((test_src.shape[0]/src_total)*100)
    
    print(f"""          Breakdown of class splits at {dir_path}
          ---------------------------------------------------
          Size of classes: {total}
          Train percentage: {train_perc}%
          Test percentage: {test_perc}%
          Train source percentage: {src_train_perc}%
          Test source percentage: {src_test_perc}%
          ==================================================""")
    
def validate_ano_splits(dir_path):
    """Validates the splits created for each anomalous class.

    Args:
        dir_path (str): The path to the root directory for the class.
    """
    X = np.load(dir_path + "X.npy")
    src = np.load(dir_path + "src.npy")
    ano_train_X = np.load(dir_path + "ano_train_X.npy")
    ano_test_X = np.load(dir_path + "ano_test_X.npy")
    ano_train_src = np.load(dir_path + "ano_train_src.npy")
    ano_test_src = np.load(dir_path + "ano_test_src.npy")
    
    total = X.shape[0]
    ano_train_perc = int((ano_train_X.shape[0]/total)*100)
    ano_test_perc = int((ano_test_X.shape[0]/total)*100)
    
    src_total = src.shape[0]
    src_train_perc = int((ano_train_src.shape[0]/src_total)*100)
    src_test_perc = int((ano_test_src.shape[0]/src_total)*100)

    print(f"""          Breakdown of class splits at {dir_path}
          ---------------------------------------------------
          Size of classes: {total}
          Anomaly train percentage: {ano_train_perc}%
          Anomaly test percentage: {ano_test_perc}%
          Anomaly train source percentage: {src_train_perc}%
          Anomaly test source percentage: {src_test_perc}%
          ==================================================""")

if __name__ == "__main__":
    normal_classes = [10, 20]
    for c in normal_classes:
        dir_path = f"data/FRGADB_Numpy/{c}/"
        create_normal_splits(dir_path)
    
    ano_classes = [40, 50]
    for c in ano_classes:
        dir_path = f"data/FRGADB_Numpy/{c}/"
        create_ano_splits(dir_path)
    
    # Sanity check for splits
    for c in normal_classes:
        dir_path = f"data/FRGADB_Numpy/{c}/"
        validate_normal_splits(dir_path)
        
    for c in ano_classes:
        dir_path = f"data/FRGADB_Numpy/{c}/"
        validate_ano_splits(dir_path)