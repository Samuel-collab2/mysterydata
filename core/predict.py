from os import path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from core.constants import OUTPUT_DIR, DATASET_INDEX_NAME, DATASET_LABEL_NAME
from core.constants_feature_set import SIGNIFICANT_RIDGE_COLUMNS, SIGNIFICANT_FORWARD_STEPWISE_COLUMNS
from core.constants_submission import SUBMISSION1_RIDGE_FEATURE_SET_COUNTS, SUBMISSION1_PROPAGATION_FEATURE_SET_COUNTS, \
    SUBMISSION2_MODEL_SETS, SUBMISSION3_MODEL_SETS, SUBMISSION4_MODEL_SETS
from core.loader import load_test_dataset
from core.model_set import ModelSet, enumerate_model_set_predictions
from core.model_set_modifiers import modifier_filter_columns


def _get_submission_dataset():
    return load_test_dataset()

def output_predictions(predictions, filename):
    """
    Outputs the predictions for a trained composite model from the test samples.
    :param predictions: The predictions
    :param filename: The name of file to write
    :return:
    """
    output = pd.DataFrame({
        DATASET_INDEX_NAME: range(len(predictions)),
        DATASET_LABEL_NAME: predictions,
    }).set_index(DATASET_INDEX_NAME)
    filepath = path.join(OUTPUT_DIR, f'{filename}.csv')
    output.to_csv(filepath)
    print(f'Wrote predictions to file: {filepath}')

def _get_submission1_model_sets(prefix, feature_set_columns, feature_set_counts):
    feature_sets = [
        feature_set_columns[:feature_count]
        for feature_count
        in feature_set_counts
    ]

    return [
        ModelSet(
            name=f'{prefix} set{index}',
            regression_modifiers=[
                modifier_filter_columns(feature_set),
            ],
        ) for index, feature_set in enumerate(feature_sets)
    ]

def _predict_model_sets(train_dataset, prefix, model_sets):
    test_dataset = _get_submission_dataset()

    for index, evaluation_predictions, evaluation_label, test_predictions in enumerate_model_set_predictions(
        train_dataset,
        test_dataset,
        model_sets
    ):
        training_mae = mean_absolute_error(
            evaluation_label,
            evaluation_predictions
        )

        print(f'Training MAE: {training_mae}')

        output_predictions(
            test_predictions,
            f'{prefix}_set{index}',
        )

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
    _predict_model_sets(dataset, 'submission1_propagation', model_sets)

def predict_submission2(dataset):
    _predict_model_sets(dataset, "submission2", SUBMISSION2_MODEL_SETS)

def predict_submission3(dataset):
    _predict_model_sets(dataset, "submission3", SUBMISSION3_MODEL_SETS)

def predict_submission4(dataset):
    _predict_model_sets(dataset, "submission4", SUBMISSION4_MODEL_SETS)
