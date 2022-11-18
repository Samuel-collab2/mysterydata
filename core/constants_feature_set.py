ALL_RAW_FEATURES = [
    'feature1',
    'feature2',
    'feature3',
    'feature4',
    'feature5',
    'feature6',
    'feature7',
    'feature8',
    'feature9',
    'feature10',
    'feature11',
    'feature12',
    'feature13',
    'feature14',
    'feature15',
    'feature16',
    'feature17',
    'feature18',
]

ALL_EXPANDED_FEATURES = [
    'feature1',
    'feature2',
    'feature3_0', 'feature3_1', 'feature3_2', 'feature3_3', 'feature3_4', 'feature3_5', 'feature3_6', 'feature3_7', 'feature3_8',
    'feature4_0', 'feature4_1',
    'feature5_0', 'feature5_1',
    'feature6',
    'feature7_0', 'feature7_1', 'feature7_2', 'feature7_3',
    'feature8',
    'feature9_0', 'feature9_1',
    'feature10',
    'feature11_0', 'feature11_1', 'feature11_2', 'feature11_3', 'feature11_4', 'feature11_5', 'feature11_6', 'feature11_7', 'feature11_8',
    'feature12',
    'feature13_0', 'feature13_1', 'feature13_2', 'feature13_3', 'feature13_4', 'feature13_5',
    'feature14_0', 'feature14_1', 'feature14_2', 'feature14_3',
    'feature15_0', 'feature15_1', 'feature15_2', 'feature15_3', 'feature15_4', 'feature15_5', 'feature15_6', 'feature15_7', 'feature15_8', 'feature15_9',
    'feature16_0', 'feature16_1', 'feature16_2', 'feature16_3', 'feature16_4', 'feature16_5',
    'feature17',
    'feature18_0', 'feature18_1', 'feature18_2'
]

SIGNIFICANT_RIDGE_COLUMNS = [
    'feature1',
    'feature3_8',
    'feature15_6',
    'feature10',
    # 'feature14_3',
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

SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS = [
    'feature1',
    'feature1*feature2',
    'feature1/feature3',
    'feature1*feature4',
    'feature1/feature5',
    'feature1*feature6',
    'feature1*feature7',
    'feature1*feature8',
    'feature1/feature9',
    'feature1*feature10',
    'feature1*feature11',
    'feature1*feature12',
    'feature1*feature13',
    'feature1*feature14',
    'feature1*feature15',
    'feature1*feature16',
    'feature1*feature17',
    'feature1*feature18',
    'feature3/feature5',
]

SIGNIFICANT_AUGMENTED_POSITIVE_REGRESSION_COLUMNS = [
    'feature15/feature16',
    'feature2/feature16',
    'feature12/feature18',
    'feature4/feature16',
    'feature8/feature16',
    'feature15/feature18',
    'feature14/feature16',
    'feature2/feature7',
    'feature2/feature6',
    'feature12/feature13',
    'feature15',
    'feature12/feature16',
    'feature12/feature14',
    'feature13/feature16',
    'feature12',
    'feature2/feature13',
    'feature2*feature15',
]

SIGNIFICANT_AUGMENTED_NEGATIVE_REGRESSION_COLUMNS = [
    'feature10/feature16',
    'feature2/feature3',
    'feature9/feature16',
    'feature17/feature18',
    'feature5/feature16',
    'feature10/feature13',
]

SIGNIFICANT_AUGMENTED_REGRESSION_COLUMNS = (
    SIGNIFICANT_AUGMENTED_POSITIVE_REGRESSION_COLUMNS
    + SIGNIFICANT_AUGMENTED_NEGATIVE_REGRESSION_COLUMNS
)
