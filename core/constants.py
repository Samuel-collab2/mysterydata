from os.path import join
import numpy as np

DATASET_TRAIN_PATH = 'trainingset.csv'
DATASET_TEST_PATH = 'testset.csv'
DATASET_COMPETITION_PATH = 'competitionset.csv'
DATASET_LABEL_NAME = 'ClaimAmount'
DATASET_INDEX_NAME = 'rowIndex'
OUTPUT_DIR = 'dist'
MODEL_PATH = join(OUTPUT_DIR, 'model.sav')

DATASET_TRAIN_RATIO = 0.8
MIN_REAL_FEATURE_UNIQUE_VALUES = 20

# Currently, increasing degrees above 2 significantly affects training time,
# because there are too many feature dimensions,
# once reduced we can evaluate if higher degree models perform better.
ANALYSIS_CROSS_VALIDATION_SETS = 5
ANALYSIS_POLYNOMIAL_DEGREES = 20
ANALYSIS_LASSO_LAMBDAS = [pow(10, exponent) for exponent in np.arange(-5, 1, 0.25)]
ANALYSIS_RIDGE_LAMBDAS = [pow(10, exponent) for exponent in range(-5, 10)]
ANALYSIS_SVC_PENALTIES = [pow(10, exponent) for exponent in range(-4, 4)]
ANALYSIS_SIGNIFICANT_FEATURE_COUNT = 20
ANALYSIS_CORRELATION_THRESHOLD = 0.45

SIGNIFICANT_FEATURE_SET_COUNTS = [1, 3, 5, 10, 15, 20]

# list of features/combinations that yield a label correlation >0.1

MENU_EXIT = ('Exit', lambda: True)
MENU_RETURN = ('<<< Back', lambda: True)
