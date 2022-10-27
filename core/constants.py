DATASET_TRAIN_PATH = 'trainingset.csv'
DATASET_TEST_PATH = 'testset.csv'
DATASET_LABEL_NAME = 'ClaimAmount'

DATASET_TRAIN_RATIO = 0.8
MIN_REAL_FEATURE_UNIQUE_VALUES = 20
CROSS_VALIDATION_SETS = 5

# Currently, increasing degrees above 2 significantly affects training time,
# because there are too many feature dimensions,
# once reduced we can evaluate if higher degree models perform better.
POLYNOMIAL_ANALYSIS_DEGREES = 2


OUTPUT_DIR = 'dist'
