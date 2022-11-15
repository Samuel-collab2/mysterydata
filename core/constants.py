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

SIGNIFICANT_FEATURE_SET_COUNTS = [1, 3, 5, 10, 15, 20]

SIGNIFICANT_RIDGE_COLUMNS = [
    'feature1',
    'feature3_8',
    'feature15_6',
    'feature10',
    'feature14_3',
    'feature14_1',
    'feature3_1',
    'feature15_5',
    'feature11_4',
    'feature16_5',
    'feature15_4',
    'feature11_3',
    'feature11_5',
    'feature8',
    'feature16_4',
    'feature17',
    'feature12',
    'feature11_7',
    'feature15_2',
    'feature3_2',
    'feature3_5',
]

SIGNIFICANT_BINARY_LABEL_COLUMNS = [
    'feature1',
    'feature3_8',
    'feature10',
    'feature7_2',
    'feature7_0',
    'feature3_2',
    'feature3_3',
    'feature12',
    'feature3_4',
    'feature16_5',
    'feature3_7',
    'feature13_1',
    'feature15_3',
    'feature16_1',
    'feature11_7',
    'feature18_1',
    'feature9_0',
    'feature9_1',
    'feature15_6',
    'feature11_3',
    'feature11_8',
]

SIGNIFICANT_FORWARD_STEPWISE_COLUMNS = [
    'feature1',
    'feature15_4',
    'feature3_8',
    'feature17',
    'feature15_6',
    'feature13_2',
    'feature15_1',
    'feature18_1',
    'feature7_2',
    'feature7_3',
    'feature9_0',
    'feature15_8',
    'feature15_2',
    'feature11_6',
    'feature3_4',
    'feature3_6',
    'feature10',
    'feature13_3',
    'feature9_1',
    'feature13_0',
]

# list of features/combinations that yield a label correlation >0.1
SIGNIFICANT_AUGMENTED_COLUMNS = [
    'feature1',
    'feature1/feature4',
    'feature1/feature5',
    'feature1/feature6',
    'feature1/feature7',
    'feature1/feature8',
    'feature1/feature10',
    'feature1/feature18',
    'feature1*feature6',
    'feature1*feature8',
    'feature1*feature10',
    'feature1*feature11',
    'feature1*feature12',
    'feature1*feature13',
    'feature1*feature15',
    'feature1*feature17',
]

MENU_EXIT = ('Exit', lambda: True)
MENU_RETURN = ('<<< Back', lambda: True)
