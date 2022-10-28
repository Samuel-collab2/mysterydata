import numpy as np

DATASET_TRAIN_PATH = 'trainingset.csv'
DATASET_TEST_PATH = 'testset.csv'
DATASET_LABEL_NAME = 'ClaimAmount'
OUTPUT_DIR = 'dist'

DATASET_TRAIN_RATIO = 0.8
MIN_REAL_FEATURE_UNIQUE_VALUES = 20
CROSS_VALIDATION_SETS = 5

# Currently, increasing degrees above 2 significantly affects training time,
# because there are too many feature dimensions,
# once reduced we can evaluate if higher degree models perform better.
ANALYSIS_POLYNOMIAL_DEGREES = 2
ANALYSIS_LASSO_LAMBDAS = [pow(10, exponent) for exponent in np.arange(-5, 10, 0.25)]
ANALYSIS_RIDGE_LAMBDAS = [pow(10, exponent) for exponent in range(-5, 10)]

MENU_EXIT = ('Exit', lambda: True)
MENU_RETURN = ('<<< Back', lambda: True)
