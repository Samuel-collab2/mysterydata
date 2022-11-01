import pandas as pd

from core.constants import SUBMISSION_SAMPLE_COUNT, DATASET_LABEL_NAME
from core.loader import load_test_dataset
from core.model_composite import train_composite
from core.preprocessing import preprocess_induction_data, separate_features_label, \
    split_claims_accept_reject, expand_dataset

SIGNIFICANT_RIDGE_FEATURES = [
    "feature1",
    "feature3_8",
    "feature15_6",
]

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

    regression_train_features, regression_train_labels = separate_features_label(regression_train_dataset, DATASET_LABEL_NAME)
    regression_test_features, _ = separate_features_label(regression_test_dataset, DATASET_LABEL_NAME)

    regression_data, _ = split_claims_accept_reject(regression_train_features, regression_train_labels)
    regression_train_features, regression_train_label = regression_data

    return induction_train_features, induction_train_label, raw_test_features, \
        regression_train_features, regression_train_label, regression_test_features

def _output_predictions(model, induction_test_features, regression_test_features):
    predictions = model.predict(induction_test_features, regression_test_features)
    for prediction, row in zip(predictions, induction_test_features["rowIndex"]):

        print(f"{row}: {prediction}")

def predict_submission1_ridge(dataset):
    induction_train_features, induction_train_label, raw_test_features, \
        regression_train_features, regression_train_label, regression_test_features = _get_submission_data(dataset)

    model = train_composite(
        induction_train_features,
        induction_train_label,
        regression_train_features.loc[:, SIGNIFICANT_RIDGE_FEATURES],
        regression_train_label,
        'submission1_ridge.json',
    )

    _output_predictions(model, raw_test_features, regression_test_features.loc[:, SIGNIFICANT_RIDGE_FEATURES])

def predict_submission1_propagation(dataset):
    induction_train_features, induction_train_label, raw_test_features, \
        regression_train_features, regression_train_label, regression_test_features = _get_submission_data(dataset)

    model = train_composite(
        induction_train_features,
        induction_train_label,
        regression_train_features.loc[:, SIGNIFICANT_PROPAGATION_FEATURES],
        regression_train_label,
        'submission1_propagation.json',
    )

    _output_predictions(model, raw_test_features, regression_test_features.loc[:, SIGNIFICANT_PROPAGATION_FEATURES])
