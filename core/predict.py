from os import path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from core.constants import SUBMISSION_SAMPLE_COUNT, DATASET_LABEL_NAME, OUTPUT_DIR
from core.loader import load_test_dataset
from core.model_composite import train_composite
from core.preprocessing import preprocess_induction_data, separate_features_label, \
    split_claims_accept_reject, expand_dataset

SIGNIFICANT_RIDGE_COUNTS = [1, 3, 5, 7, 10]
SIGNIFICANT_RIDGE_FEATURES = [
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
    'feature15_4'
]

SIGNIFICANT_PROPAGATION_COUNTS = [3, 5, 10, 15, 20]
SIGNIFICANT_PROPAGATION_FEATURES = [
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
    'feature13_0'
]

def _get_submission_features():
    dataset = load_test_dataset()
    return dataset.sample(n=SUBMISSION_SAMPLE_COUNT)

def _get_submission_data(dataset):
    raw_train_dataset = dataset
    raw_test_features = _get_submission_features()

    raw_features, raw_label = separate_features_label(raw_train_dataset, DATASET_LABEL_NAME)
    induction_train_features, induction_train_label = preprocess_induction_data(raw_features, raw_label)

    combined_dataset = pd.concat([raw_train_dataset, raw_test_features], axis=0)
    combined_expanded_dataset = expand_dataset(combined_dataset)

    train_dataset_length = len(raw_train_dataset)
    regression_train_dataset = combined_expanded_dataset.iloc[:train_dataset_length]
    regression_test_dataset = combined_expanded_dataset.iloc[train_dataset_length:]

    regression_train_features, regression_train_labels = separate_features_label(
        regression_train_dataset,
        DATASET_LABEL_NAME
    )

    regression_test_features, _ = separate_features_label(
        regression_test_dataset,
        DATASET_LABEL_NAME
    )

    regression_data, _ = split_claims_accept_reject(regression_train_features, regression_train_labels)
    regression_train_features, regression_train_label = regression_data

    return induction_train_features, induction_train_label, raw_test_features, \
        regression_train_features, regression_train_label, regression_test_features

def _output_predictions(model, induction_test_features, regression_test_features, filename):
    predictions = model.predict(induction_test_features, regression_test_features)
    output = pd.DataFrame({
        'rowIndex': induction_test_features['rowIndex'].values,
        'ClaimAmount': predictions,
    }).set_index('rowIndex').sort_index()
    filepath = path.join(OUTPUT_DIR, f'{filename}.csv')
    output.to_csv(filepath)
    print(f'Wrote predictions to file: {filepath}')

def _predict_feature_sets(dataset, feature_sets, file_prefix):
    induction_train_features, induction_train_label, raw_test_features, \
        regression_train_features, regression_train_label, regression_test_features = _get_submission_data(dataset)

    for index, feature_set in enumerate(feature_sets):
        filename = f'{file_prefix}_set{index}'

        print(f'Predicting {filename}...')
        print(feature_set)

        model = train_composite(
            induction_train_features,
            induction_train_label,
            regression_train_features.loc[:, feature_set],
            regression_train_label,
            file_prefix,
        )

        # training_predictions = model.predict(induction_train_features, regression_train_features.loc[:, feature_set])
        # training_mae = mean_absolute_error(regression_train_label, training_predictions)
        # print(f'Training MAE: {training_mae}')

        _output_predictions(model, raw_test_features, regression_test_features.loc[:, feature_set], filename)

def predict_submission1_ridge(dataset):
    feature_sets = [
        SIGNIFICANT_RIDGE_FEATURES[:feature_count]
        for feature_count
        in SIGNIFICANT_RIDGE_COUNTS
    ]
    _predict_feature_sets(dataset, feature_sets, 'submission1_ridge')

def predict_submission1_propagation(dataset):
    feature_sets = [
        SIGNIFICANT_PROPAGATION_FEATURES[:feature_count]
        for feature_count
        in SIGNIFICANT_PROPAGATION_COUNTS
    ]
    _predict_feature_sets(dataset, feature_sets, 'submission1_propagation')
