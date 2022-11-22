import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.constants import DATASET_TRAIN_PATH, DATASET_TEST_PATH, DATASET_COMPETITION_PATH


def load_train_dataset():
    return _read_dataset_file(DATASET_TRAIN_PATH)

def load_standardized_train_dataset():
    return _standardize_dataset(load_train_dataset())

def load_test_dataset():
    return _read_dataset_file(DATASET_TEST_PATH)

def load_determining_dataset():
    return load_train_dataset()

def load_competition_dataset():
    return _read_dataset_file(DATASET_COMPETITION_PATH)

def _read_dataset_file(filepath):
    return pd.read_csv(filepath, low_memory=False)

def _standardize_dataset(dataset):
    return pd.DataFrame(
        StandardScaler().fit_transform(dataset),
        columns=dataset.columns
    )
