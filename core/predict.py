from os import path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from core.constants import DATASET_LABEL_NAME, OUTPUT_DIR, SIGNIFICANT_RIDGE_COLUMNS, \
    SIGNIFICANT_FORWARD_STEPWISE_COLUMNS
from core.constants_submission import SUBMISSION1_RIDGE_FEATURE_SET_COUNTS, SUBMISSION1_PROPAGATION_FEATURE_SET_COUNTS, \
    SUBMISSION2_MODEL_SETS
from core.loader import load_test_dataset, load_determining_dataset
from core.model_composite import train_composite
from core.model_induction import train_classifier_tree
from core.model_regression import train_linear_regression
from core.modifiers import modifier_filter_columns
from core.preprocessing import separate_features_label, \
    split_claims_accept_reject, expand_dataset_deterministic, get_categorical_columns, convert_label_boolean

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

    induction_train_label = convert_label_boolean(expanded_train_label)

    regression_test_features, _ = separate_features_label(
        expanded_test_dataset,
        DATASET_LABEL_NAME
    )

    accept_data, _ = split_claims_accept_reject(expanded_train_features, expanded_train_label)
    accept_train_features, accept_train_label = accept_data

    return (expanded_train_features, induction_train_label, expanded_test_features, expanded_train_features), \
        (accept_train_features, accept_train_label, expanded_test_features, expanded_train_features), \
        expanded_train_label

def _output_predictions(model, induction_test_features, regression_test_features, filename):
    predictions = model.predict(induction_test_features, regression_test_features)
    output = pd.DataFrame({
        'rowIndex': induction_test_features.index,
        'ClaimAmount': predictions,
    }).set_index('rowIndex').sort_index()
    filepath = path.join(OUTPUT_DIR, f'{filename}.csv')
    output.to_csv(filepath)
    print(f'Wrote predictions to file: {filepath}')

def _predict_model_sets(dataset, prefix, model_sets):
    submission_data = _get_submission_data(dataset)

    for index, (name, (
        train_induction_model,
        induction_modifiers,
        train_regression_model,
        regression_modifiers
    )) in enumerate(model_sets.items()):

        print(f'Predicting: {name}...')

        induction_data, regression_data, evaluation_label = submission_data
        for induction_modifier in induction_modifiers:
            induction_data = induction_modifier(*induction_data)

        for regression_modifier in regression_modifiers:
            regression_data = regression_modifier(*regression_data)

        induction_train_features, induction_train_label, \
            induction_test_features, induction_evaluation_features = induction_data

        regression_train_features, regression_train_label, \
            regression_test_features, regression_evaluation_features = regression_data

        model = train_composite(
            induction_train_features,
            induction_train_label,
            regression_train_features,
            regression_train_label,
            train_induction_model,
            train_regression_model
        )

        training_predictions = model.predict(
            induction_evaluation_features,
            regression_evaluation_features
        )

        training_mae = mean_absolute_error(
            evaluation_label,
            training_predictions
        )

        print(f'Training MAE: {training_mae}')

        _output_predictions(
            model,
            induction_test_features,
            regression_test_features,
            f'{prefix}_set{index}'
        )

def _get_submission1_model_sets(prefix, feature_set_columns, feature_set_counts):
    feature_sets = [
        feature_set_columns[:feature_count]
        for feature_count
        in feature_set_counts
    ]

    return {
        f'{prefix} set{index}': (
            train_classifier_tree, [],
            train_linear_regression, [
                modifier_filter_columns(feature_set),
            ]) for index, feature_set
        in enumerate(feature_sets)
    }

def predict_submission1_ridge(dataset):
    model_sets = _get_submission1_model_sets(
        "Ridge",
        SIGNIFICANT_RIDGE_COLUMNS,
        SUBMISSION1_RIDGE_FEATURE_SET_COUNTS
    )
    _predict_model_sets(dataset, 'submission1_ridge', model_sets)

def predict_submission1_propagation(dataset):
    model_sets = _get_submission1_model_sets(
        "Propagation",
        SIGNIFICANT_FORWARD_STEPWISE_COLUMNS,
        SUBMISSION1_PROPAGATION_FEATURE_SET_COUNTS
    )
    _predict_model_sets(dataset, 'submission1_ridge', model_sets)

def predict_submission2(dataset):
    _predict_model_sets(dataset, "submission2", SUBMISSION2_MODEL_SETS)
