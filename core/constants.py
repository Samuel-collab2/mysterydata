import numpy as np

DATASET_TRAIN_PATH = 'trainingset.csv'
DATASET_TEST_PATH = 'testset.csv'
DATASET_LABEL_NAME = 'ClaimAmount'
OUTPUT_DIR = 'dist'

DATASET_TRAIN_RATIO = 0.8
MIN_REAL_FEATURE_UNIQUE_VALUES = 20

# Currently, increasing degrees above 2 significantly affects training time,
# because there are too many feature dimensions,
# once reduced we can evaluate if higher degree models perform better.
ANALYSIS_CROSS_VALIDATION_SETS = 5
ANALYSIS_POLYNOMIAL_DEGREES = 20
ANALYSIS_LASSO_LAMBDAS = [pow(10, exponent) for exponent in np.arange(-5, 1, 0.25)]
ANALYSIS_RIDGE_LAMBDAS = [pow(10, exponent) for exponent in range(-5, 10)]
ANALYSIS_SIGNIFICANT_FEATURE_COUNT = 20
ANALYSIS_CORRELATION_THRESHOLD = 0.45

MENU_EXIT = ('Exit', lambda: True)
MENU_RETURN = ('<<< Back', lambda: True)

SIGNIFICANT_REGRESSION_FEATURE_COUNT = 3
SIGNIFICANT_REGRESSION_FEATURES = [
    "feature1",
    "feature3_8",
    "feature15_6",
    "feature10",
    "feature14_3",
    "feature14_1",
    "feature3_1",
    "feature15_5",
    "feature11_4",
    "feature16_5",
    "feature15_4",
]
