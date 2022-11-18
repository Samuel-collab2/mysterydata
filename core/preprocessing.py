import pandas as pd
from sklearn.model_selection import train_test_split
from core.constants import MIN_REAL_FEATURE_UNIQUE_VALUES, DATASET_LABEL_NAME
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS

def separate_features_label(dataset, label_column):
    return (
        dataset.drop(label_column, axis='columns'),
        dataset.loc[:, label_column],
    )

def split_training_test(features, label, train_factor, shuffle=False, seed=None):
    train_features, test_features, train_label, test_label = train_test_split(
        features,
        label,
        train_size=train_factor,
        shuffle=shuffle,
        random_state=seed,
    )
    return (train_features, train_label), (test_features, test_label)

def split_claims_accept_reject(features, label):
    accept_indices = label[label != 0].index
    reject_indices = label[label == 0].index
    return (features.iloc[accept_indices], label.iloc[accept_indices]), (features.iloc[reject_indices], label.iloc[reject_indices])

def convert_label_binary(label):
    """
    Converts label values that are greater than 0 to 1.
    :param label: The label
    :return: The converted label
    """
    return label.mask((label > 0) | (label < 0), 1)

def convert_label_boolean(label):
    """
    Converts label values to true and false.
    :param label: The label
    :return: The converted label
    """
    return pd.Series(label).map(bool)

def is_categorical_feature(column, unique_occurrence_threshold):
    return (
        column.dtype == 'object'
        or column.nunique() < unique_occurrence_threshold
    )

def get_categorical_columns(dataset):
    """
    Gets categorical columns from a dataset based on evaluation from the `is_categorical_feature(...)` function.
    :param dataset: The dataset
    :return: The categorical column names
    """
    return [
        column for column
        in dataset.columns
        if is_categorical_feature(dataset[column], MIN_REAL_FEATURE_UNIQUE_VALUES)
    ]

def encode_feature(feature):
    """
    Encodes a single feature by separating it into a multiple columns based on its values.
    :param feature: The feature to encode
    :return: The encoded features based on given feature
    """

    return pd.get_dummies(
        feature,
        columns=[feature.name],
    )

def expand_dataset_deterministic(raw_dataset, determining_dataset, expanded_columns):
    """
    Expands the dataset making sure it would have the same column names
    as the determining dataset.

    :param expanded_columns: The columns to expand
    :param raw_dataset: The dataset to expand
    :param determining_dataset: The dataset to reference column names
    :return: The expanded dataset
    """

    expand_raw = expand_dataset(raw_dataset, expanded_columns)
    expand_determining = expand_dataset(determining_dataset, expanded_columns)

    return expand_raw.rename({
        old_column: new_column
        for old_column, new_column
        in zip(expand_raw.columns, expand_determining.columns)
    })

def expand_dataset(raw_dataset, expanded_columns):
    """
    Expands the dataset based on given categorical columns.

    :param expanded_columns: The columns to expand
    :param raw_dataset: The dataset to expand
    :return: The expanded dataset
    """

    subsets = []
    for column in raw_dataset.columns:
        feature = raw_dataset[column]
        is_expanded = column in expanded_columns
        if is_expanded:
            subset = encode_feature(feature)
            subset_columns = {subset_column: f'{column}_{index}' for index, subset_column in enumerate(subset.columns)}
            subsets.append(subset.rename(columns=subset_columns))
        else:
            subsets.append(feature)

    return pd.concat(subsets, axis=1)

def balance_binary_dataset(train_features, train_labels, skew_true=1, skew_false=1):
    """
    Balances a binary dataset (i.e. boolean labels).
    Skew paramaters are used to fine-tune bias.
    :param train_features: Training features
    :param train_labels: Training labels
    :param skew_true: Factor of true sample count in resulting dataset
    :param skew_false: Factor of false sample count in resulting dataset
    """

    dataset_label_name = train_labels.name

    train_samples = pd.concat((train_features, train_labels), axis='columns')

    true_samples = train_samples[train_samples[dataset_label_name] == True]
    false_samples = train_samples[train_samples[dataset_label_name] == False]

    min_samples = min(len(true_samples), len(false_samples))

    true_samples = true_samples[:min_samples * skew_true]
    false_samples = false_samples[:min_samples * skew_false]
    train_samples = pd.concat((true_samples, false_samples))

    train_features, train_labels = separate_features_label(train_samples, dataset_label_name)
    return train_features, train_labels

def create_augmented_features(features, feature_subset):
    """
    Augments an existing feature dataset.
    An augmented feature models a relationship between two features.
    Atomic features are not included in the resulting matrix.
    :param features: a data frame containing the features to augment
    :param feature_subset: a list of augmented feature names to parse
                           each feature name supports multiplication (*) and division (/)
    :return: the augmented data frame
    """

    product_pairs = [pattern.split('*')
        for pattern in feature_subset
            if '*' in pattern]
    quotient_pairs = [pattern.split('/')
        for pattern in feature_subset
            if '/' in pattern]

    feature_map = {
        f'{feature1}*{feature2}': features[feature1] * features[feature2]
            for feature1, feature2 in product_pairs
    } | {
        f'{feature1}/{feature2}': features[feature1] / features[feature2]
            for feature1, feature2 in quotient_pairs
    }

    feature_names, feature_data = zip(*feature_map.items())
    features_augmented = pd.concat(feature_data, axis='columns')
    features_augmented.columns = feature_names

    # HACK: numpy handles zero division with infs
    features_augmented.replace(float('inf'), 0, inplace=True)
    features_augmented.fillna(0, inplace=True)
    return features_augmented

def get_induction_data(train_data, test_data):
    train_features, train_labels = separate_features_label(train_data, DATASET_LABEL_NAME)
    test_features = test_data

    categorical_columns = get_categorical_columns(train_features)
    categorical_train_features = train_features.loc[:, categorical_columns]
    categorical_test_features = test_features.loc[:, categorical_columns]

    combined_features = pd.concat((train_features, test_features))
    combined_expanded_features = expand_dataset_deterministic(
        combined_features,
        train_features,
        categorical_columns,
    )

    train_size = len(train_features)
    expanded_train_features = combined_expanded_features.iloc[:train_size]
    expanded_test_features = combined_expanded_features.iloc[train_size:]

    augmented_train_features = create_augmented_features(train_features, SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS)
    augmented_test_features = create_augmented_features(test_features, SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS)

    processed_train_features = pd.concat((
        expanded_train_features,
        augmented_train_features,
        categorical_train_features,
    ), axis='columns')

    processed_test_features = pd.concat((
        expanded_test_features,
        augmented_test_features,
        categorical_test_features,
    ), axis='columns')

    processed_train_labels = convert_label_boolean(train_labels)
    return (
        processed_train_features,
        processed_train_labels,
        processed_test_features,
    )
