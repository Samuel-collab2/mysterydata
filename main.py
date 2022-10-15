from os import mkdir

import numpy as np
import pandas as pd


TRAIN_DATASET_PATH = 'trainingset.csv'
TEST_DATASET_PATH = 'testset.csv'
OUTPUT_DIR = 'dist'
try:
    mkdir(OUTPUT_DIR)
except OSError:
    pass


def main():
    pass

if __name__ == '__main__':
    main()
