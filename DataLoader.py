import numpy as np
import pickle
import tarfile

"""This script implements the functions for reading data.
"""

# data_dir="C:/Users/Downloads/cifar-10-python.tar.gz"

def unzip(file):
    """Extracts a tar file.
    Args:
        file_path: A string. The path to the tar file.
    Returns:
        tar: TarFile object after extraction.
    """
    tar = tarfile.open(file, "r")
    tar.extractall()
    
    return tar


def unpickle(file):
    """Loads a pickle file.
    Args:
        file_path: A string. The path to the pickle file.
    Returns:
        Loaded pickle object.
    """
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    
    return dict


def load_data(data_dir):
    """Load the CIFAR-10 dataset.
    Args:
        data_dir: A string. The directory where data batches
            are stored.
    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """
    match = ["batch_1", "batch_2", "batch_3", "batch_4", "batch_5"]
    j = 0
    files = unzip(data_dir)
    for i in files.getnames():
        if any(x in i for x in match):
            if j == 0:
                x_train = unpickle(i)[b"data"]
                y_train = unpickle(i)[b"labels"]
                j = j + 1
            else:
                x_train = np.concatenate((x_train, unpickle(i)[b"data"]))
                y_train = np.concatenate((y_train, unpickle(i)[b"labels"]))
        if ("test") in i:
            x_test = unpickle(i)[b"data"]
            y_test = unpickle(i)[b"labels"]
            y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.
    Args:
        data_dir: A string. The directory where the testing images
        are stored.
    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """
    x_test = np.load(data_dir)

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.
    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[: int(train_ratio * x_train.shape[0])]
    y_train_new = y_train[: int(train_ratio * x_train.shape[0])]
    x_valid = x_train[int(train_ratio * x_train.shape[0]) :]
    y_valid = y_train[int(train_ratio * x_train.shape[0]) :]

    return x_train_new, y_train_new, x_valid, y_valid
