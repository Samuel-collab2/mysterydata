import pandas as pd

from core.preprocessing import separate_features_label, get_categorical_columns, \
    expand_dataset_deterministic, convert_label_boolean, create_augmented_features
from core.constants import DATASET_LABEL_NAME
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_COLUMNS


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

    augmented_train_features = create_augmented_features(train_features, SIGNIFICANT_AUGMENTED_COLUMNS)
    augmented_test_features = create_augmented_features(test_features, SIGNIFICANT_AUGMENTED_COLUMNS)

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


def sandbox_iterative_induction(train_data, test_data):
    train_features, train_labels, test_features = get_induction_data(train_data, test_data)
    print(train_features.columns)


if __name__ == '__main__':
    from core.loader import load_train_dataset, load_test_dataset
    print('Loading data for induction test suite...')
    sandbox_iterative_induction(
        train_data=load_train_dataset(),
        test_data=load_test_dataset(),
    )
