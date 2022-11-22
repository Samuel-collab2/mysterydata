from dataclasses import dataclass, field

import pandas as pd

from core.constants import DATASET_LABEL_NAME
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_COLUMNS
from core.loader import load_determining_dataset
from core.model_composite import train_composite
from core.model_induction import train_random_forest
from core.model_regression import train_linear_regression
from core.model_set_modifiers import modifier_filter_expanded
from core.preprocessing import separate_features_label, get_categorical_columns, expand_dataset_deterministic, \
    create_augmented_features, convert_label_boolean, split_claims_accept_reject


@dataclass(frozen=True)
class ModelSet:
    name: str
    train_induction_model: callable = field(default=train_random_forest)
    induction_modifiers: list = field(default_factory=lambda: [modifier_filter_expanded()])
    train_regression_model: callable = field(default=train_linear_regression)
    regression_modifiers: list = field(default_factory=lambda: [modifier_filter_expanded()])
    proba_threshold: bool = 0.5


def get_model_set_data(train_dataset, test_features):
    """
    Preprocesses the data for submission.

    Loads training and test datasets.
    Combines them for mutual dataset expansion to ensure they have the same columns.
    Splits them back into expanded training and test datasets.

    Separate features and label for each dataset.

    Converts induction label to boolean values.

    Filters accepted claims to use for regression training.

    Returns the processed data.

    :param train_dataset: The train dataset
    :param test_features: The test dataset
    :return: Tuple containing (induction_data, regression_data, evaluation_label)
    """
    raw_train_dataset = train_dataset
    raw_test_features = test_features

    raw_train_features, raw_train_label = separate_features_label(raw_train_dataset, DATASET_LABEL_NAME)

    categorical_columns = get_categorical_columns(raw_train_dataset)

    categorical_train_features = raw_train_features.loc[:, categorical_columns]
    categorical_test_features = raw_test_features.loc[:, categorical_columns]

    combined_dataset = pd.concat([raw_train_dataset, raw_test_features], axis=0)
    combined_expanded_dataset = expand_dataset_deterministic(
        combined_dataset,
        train_dataset,
        categorical_columns
    )

    train_dataset_length = len(raw_train_dataset)
    expanded_train_dataset = combined_expanded_dataset.iloc[:train_dataset_length]
    expanded_test_dataset = combined_expanded_dataset.iloc[train_dataset_length:]

    expanded_train_features, _ = separate_features_label(
        expanded_train_dataset,
        DATASET_LABEL_NAME
    )

    expanded_test_features, _ = separate_features_label(
        expanded_test_dataset,
        DATASET_LABEL_NAME
    )

    augmented_train_features = create_augmented_features(raw_train_features, SIGNIFICANT_AUGMENTED_COLUMNS)
    augmented_test_features = create_augmented_features(raw_test_features, SIGNIFICANT_AUGMENTED_COLUMNS)

    processed_train_features = pd.concat(
        (expanded_train_features, augmented_train_features, categorical_train_features),
        axis='columns'
    )

    processed_test_features = pd.concat(
        (expanded_test_features, augmented_test_features, categorical_test_features),
        axis='columns'
    )

    induction_train_label = convert_label_boolean(raw_train_label)

    accept_data, _ = split_claims_accept_reject(processed_train_features, raw_train_label)
    accept_train_features, accept_train_label = accept_data

    return (processed_train_features, induction_train_label, processed_test_features, processed_train_features), \
        (accept_train_features, accept_train_label, processed_test_features, processed_train_features), \
        raw_train_label

def get_model_set_train_data(train_dataset):
    """
    Preprocesses training data.
    :param: a data frame containing the training data
    :return: (
        (induction_features, induction_labels, _, _),
        (regression_features, regression_labels, _, _),
    )
    """
    train_features, train_labels = separate_features_label(train_dataset, DATASET_LABEL_NAME)

    categorical_columns = get_categorical_columns(train_dataset)
    categorical_train_features = train_features.loc[:, categorical_columns]
    expanded_train_features = expand_dataset_deterministic(
        train_features,
        load_determining_dataset(),
        categorical_columns
    )
    augmented_train_features = create_augmented_features(
        train_features,
        SIGNIFICANT_AUGMENTED_COLUMNS
    )

    induction_train_features = pd.concat((
        expanded_train_features,
        augmented_train_features,
        categorical_train_features,
    ), axis='columns')
    induction_train_columns = induction_train_features.columns

    induction_train_labels = convert_label_boolean(train_labels)
    induction_data = (induction_train_features, induction_train_labels,
        pd.DataFrame(columns=induction_train_columns),
        pd.DataFrame(columns=induction_train_columns))

    accept_data, _ = split_claims_accept_reject(induction_train_features, train_labels)
    accept_train_features, accept_train_label = accept_data
    regression_data = (accept_train_features, accept_train_label,
        pd.DataFrame(columns=induction_train_columns),
        pd.DataFrame(columns=induction_train_columns))

    return induction_data, regression_data

def enumerate_model_set_predictions(train_dataset, test_dataset, model_sets):
    """
    Output predictions and training MAE for the given model sets.

    A model set is an enumeration of models, model settings and data modifiers to use for predictions.
    A model set is given in the format:
    (train_induction_model, induction_modifiers, train_regression_model, regression_modifiers)

    The train functions are used to construct and train the models.
    The modifier lists are used to process the data before they go into the models.

    This allows for modular setup of complex models.

    Each model set is trained and used to output predictions to a file.

    :param train_dataset: The train dataset
    :param test_dataset: The test dataset
    :param model_sets: The model sets
    :return:
    """
    submission_data = get_model_set_data(train_dataset, test_dataset)

    for index, model_set in enumerate(model_sets):
        evaluation_predictions, evaluation_label, test_predictions = get_model_set_predictions(model_set, submission_data)
        yield index, evaluation_predictions, evaluation_label, test_predictions

def modify_model_set_train_data(model_set, induction_data, regression_data):
    for induction_modifier in model_set.induction_modifiers:
        induction_data = induction_modifier(*induction_data)

    for regression_modifier in model_set.regression_modifiers:
        regression_data = regression_modifier(*regression_data)

    return induction_data, regression_data

def train_model_from_model_set(model_set, induction_data, regression_data):
    induction_data, regression_data = modify_model_set_train_data(
        model_set,
        induction_data,
        regression_data,
    )

    induction_train_features, induction_train_label, _, _ = induction_data
    regression_train_features, regression_train_label, _, _ = regression_data

    model = train_composite(
        induction_train_features,
        induction_train_label,
        regression_train_features,
        regression_train_label,
        model_set.train_induction_model,
        model_set.train_regression_model,
        proba_threshold=model_set.proba_threshold,
    )

    return model

def get_model_set_predictions(model_set, submission_data):
    print(f'Predicting: {model_set.name}...')

    induction_data, regression_data, evaluation_label = submission_data
    model = train_model_from_model_set(model_set, induction_data, regression_data)

    induction_data, regression_data = modify_model_set_train_data(
        model_set,
        induction_data,
        regression_data,
    )

    _, _, induction_test_features, induction_evaluation_features = induction_data
    _, _, regression_test_features, regression_evaluation_features = regression_data

    evaluation_predictions = model.predict(
        induction_evaluation_features,
        regression_evaluation_features
    )

    test_predictions = model.predict(
        induction_test_features,
        regression_test_features,
    )

    return evaluation_predictions, evaluation_label, test_predictions
