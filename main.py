from os import mkdir

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_DATASET_PATH = 'trainingset.csv'
TEST_DATASET_PATH = 'testset.csv'
DATASET_LABEL = 'ClaimAmount'
TRAIN_FACTOR = 0.8
OUTPUT_DIR = 'dist'
try:
    mkdir(OUTPUT_DIR)
except OSError:
    pass


def load_dataset(file_path):
    return pd.read_csv(file_path)

def split_data(data, label, train_factor):
    x = data.drop(label, axis='columns', inplace=False)
    y = data.loc[:, label]
    return train_test_split(x, y, train_size=train_factor, shuffle=False)

def main():
    train_data = load_dataset(TRAIN_DATASET_PATH)
    data_split = split_data(train_data,
        label=DATASET_LABEL,
        train_factor=TRAIN_FACTOR)

if __name__ == '__main__':
    main()
