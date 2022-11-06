from os import path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from core.constants import DATASET_LABEL_NAME, OUTPUT_DIR, SIGNIFICANT_RIDGE_COLUMNS, \
    SIGNIFICANT_FORWARD_STEPWISE_COLUMNS, SIGNIFICANT_FEATURE_SET_COUNTS, \
    SIGNIFICANT_BINARY_LABEL_COLUMNS
from core.loader import load_test_dataset, load_determining_dataset
from core.model_composite import train_composite
from core.preprocessing import separate_features_label, \
    split_claims_accept_reject, expand_dataset_deterministic, get_categorical_columns


def _get_submission_dataset():
    return load_test_dataset()

def _get_submission_data(dataset):
    raw_train_dataset = dataset
    raw_test_features = _get_submission_dataset()

    categorical_columns = get_categorical_columns(raw_train_dataset)
    combined_dataset = pd.concat([raw_train_dataset, raw_test_features], axis=0)
    combined_expanded_dataset = expand_dataset_deterministic(
        combined_dataset,
        load_determining_dataset(),
        categorical_columns
    )

    train_dataset_length = len(raw_train_dataset)
    expanded_train_dataset = combined_expanded_dataset.iloc[:train_dataset_length]
    expanded_test_dataset = combined_expanded_dataset.iloc[train_dataset_length:]

    expanded_train_features, expanded_train_label = separate_features_label(
        expanded_train_dataset,
        DATASET_LABEL_NAME
    )

    expanded_test_features, expanded_test_label = separate_features_label(
        expanded_test_dataset,
        DATASET_LABEL_NAME
    )

    induction_train_label = pd.Series(expanded_train_label).map(bool)

    regression_test_features, _ = separate_features_label(
        expanded_test_dataset,
        DATASET_LABEL_NAME
    )

    accept_data, _ = split_claims_accept_reject(expanded_train_features, expanded_train_label)
    accept_train_features, accept_train_label = accept_data

    return expanded_train_features, induction_train_label, expanded_test_features, \
        accept_train_features, accept_train_label, expanded_test_features, \
        expanded_train_features, expanded_train_features, expanded_train_label

def _output_predictions(model, induction_test_features, regression_test_features, filename):
    predictions = model.predict(induction_test_features.drop('rowIndex', axis='columns'), regression_test_features)
    output = pd.DataFrame({
        'rowIndex': induction_test_features['rowIndex'].values,
        'ClaimAmount': predictions,
    }).set_index('rowIndex').sort_index()
    filepath = path.join(OUTPUT_DIR, f'{filename}.csv')
    output.to_csv(filepath)
    print(f'Wrote predictions to file: {filepath}')

def _predict_feature_sets(dataset, feature_sets, file_prefix):
    induction_train_features, induction_train_label, induction_test_features, \
        regression_train_features, regression_train_label, regression_test_features, \
        evaluation_induction_features, evaluation_regression_features, evaluation_label = _get_submission_data(dataset)

    induction_columns = SIGNIFICANT_BINARY_LABEL_COLUMNS

    for index, feature_set in enumerate(feature_sets):
        filename = f'{file_prefix}_set{index}'

        print(f'Predicting {filename}...')
        print(feature_set)

        model = train_composite(
            induction_train_features.loc[:, induction_columns],
            induction_train_label,
            regression_train_features.loc[:, feature_set],
            regression_train_label,
            file_prefix,
        )

        training_predictions = model.predict(
            evaluation_induction_features.loc[:, induction_columns],
            evaluation_regression_features.loc[:, feature_set]
        )

        training_mae = mean_absolute_error(
            evaluation_label,
            training_predictions
        )

        print(f'Training MAE: {training_mae}')

        _output_predictions(model,
            induction_test_features.loc[:, ('rowIndex', *induction_columns)],
            regression_test_features.loc[:, feature_set],
            filename)

def predict_submission1_ridge(dataset):
    feature_sets = [
        SIGNIFICANT_RIDGE_COLUMNS[:feature_count]
        for feature_count
        in SIGNIFICANT_FEATURE_SET_COUNTS
    ]
    _predict_feature_sets(dataset, feature_sets, 'submission1_ridge')

def predict_submission1_propagation(dataset):
    feature_sets = [
        SIGNIFICANT_FORWARD_STEPWISE_COLUMNS[:feature_count]
        for feature_count
        in SIGNIFICANT_FEATURE_SET_COUNTS
    ]
    _predict_feature_sets(dataset, feature_sets, 'submission1_propagation')
