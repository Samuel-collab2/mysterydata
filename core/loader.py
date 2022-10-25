import pandas as pd
from core.constants import DATASET_TRAIN_PATH, DATASET_TEST_PATH

def load_train_dataset():
    return _read_dataset_file(DATASET_TRAIN_PATH)

def load_test_dataset():
    return _read_dataset_file(DATASET_TEST_PATH)

def _read_dataset_file(filepath):
    return pd.read_csv(filepath, low_memory=False)
