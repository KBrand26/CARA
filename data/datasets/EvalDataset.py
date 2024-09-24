from torch.utils.data import Dataset
from torchvision import transforms

class EvalDataset(Dataset):
    """
    Very basic dataset to encapsulate our data for use in dataloader. Contains additional information that could
    be useful during evaluation.
    """
    
    def __init__(self, X, y, y_true, src):
        """Initialize the data for the dataset, along with any transforms that have to be performed.

        Args:
            X (numpy.ndarray): The array containing the image data.
            y (numpy.ndarray): The array containing the labels indicating whether an image is an anomaly.
            y_true (numpy.ndarray): The array containing the true labels of the class to which the image belonged.
            src (numpy.ndarray): The array containing the file name from which the image originates.
        """
        self.X = X
        self.y = y
        self.y_true = y_true
        self.src = src
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        """
        Determines the number of samples in the dataset
        """
        return len(self.y)
    
    def __getitem__(self, idx):
        """Returns an image and its corresponding labels from the dataset.

        Args:
            idx (int): The index from which to extract the image and labels.

        Returns:
            Tuple[numpy.ndarray, int, str]: A tuple containing the image, a label indicating whether
              the image is an anomaly and the true label of the image.
        """
        sample = self.X[idx]
        
        if self.transform:
            sample = self.transform(sample)

        return sample, self.y[idx], self.y_true[idx], str(self.src[idx])