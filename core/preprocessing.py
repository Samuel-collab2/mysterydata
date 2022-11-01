import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from core.constants import MIN_REAL_FEATURE_UNIQUE_VALUES

def zip_sort(lead_list, follow_list, comparator=lambda x: x[0], reverse=False):
    return zip(*sorted(zip(lead_list, follow_list), key=comparator, reverse=reverse))

def enumerate_cross_validation_sets(features, label, sets):
    kf = KFold(n_splits=sets)
    for train_indices, valid_indices in kf.split(features, label):
        yield features.iloc[train_indices], label.iloc[train_indices], features.iloc[valid_indices], label.iloc[valid_indices]

def separate_features_label(dataset, label_column):
    return (
        dataset.drop(label_column, axis='columns', inplace=False),
        dataset.loc[:, label_column],
    )

def split_training_test(features, label, train_factor, shuffle=False):
    train_features, test_features, train_label, test_label = train_test_split(
        features,
        label,
        train_size=train_factor,
        shuffle=shuffle,
    )
    return (train_features, train_label), (test_features, test_label)

def split_claims_accept_reject(features, label):
    accept_indices = label[label != 0].index
    reject_indices = label[label == 0].index
    return (features.iloc[accept_indices], label.iloc[accept_indices]), (features.iloc[reject_indices], label.iloc[reject_indices])

def preprocess_induction_data(features, label):
    return expand_dataset(features), pd.Series(label).map(bool)

def convert_label_binary(label):
    return label.mask(label > 0, 1, inplace=False)

def is_categorical_feature(column):
    return (column.dtype == 'object'
        or column.nunique() < MIN_REAL_FEATURE_UNIQUE_VALUES)

def encode_feature(feature):
    return pd.get_dummies(
        feature,
        columns=[feature.name],
    )

def expand_dataset_indexed(dataset, is_expanded_column=is_categorical_feature):
    subsets = []
    for column in dataset.columns:
        feature = dataset[column]
        is_encoded = is_expanded_column(feature)
        if is_encoded:
            subset = encode_feature(feature)
            subset_columns = {subset_column: f'{column}_{index}' for index, subset_column in enumerate(subset.columns)}
            subset.rename(columns=subset_columns, inplace=True)
            subsets.append(subset)
        else:
            subsets.append(feature)

    return pd.concat(subsets, axis=1)

def expand_dataset(dataset, is_expanded_column=is_categorical_feature):
    """
    Expands the dataset by splitting each categorical column into many
    :param dataset: the raw dataset
    :return: the expanded dataset
    """

    categorical_columns = [column for column in dataset.columns
                           if is_expanded_column(dataset[column])]

    data_expanded = pd.get_dummies(dataset,
                                   columns=categorical_columns,
                                   prefix=categorical_columns)

    def get_column_prefix(column):
        """
        Gets the categorical column prefix for the given column, if existent
        :param column: the column to extract the prefix from
        :return: the column prefix
        """

        if '_' not in column:
            return None

        underscore_index = len(column) - column[::-1].index('_') - 1
        column_prefix = column[:underscore_index]
        if column_prefix.endswith('_'):
            return column_prefix[:-1]

        return column_prefix

    def index_column(column):
        """
        Determines the original index for a given column (used for sorting)
        :param column: the column to index
        :return: the index, len(columns) if column could not be indexed
        """

        if column in data_columns:
            return data_columns.index(column)

        column_prefix = get_column_prefix(column)
        if column_prefix is None:
            return len(data_columns)

        return data_columns.index(column_prefix)

    data_columns = list(dataset.columns)
    return data_expanded.reindex(
        sorted(data_expanded.columns,
        key=index_column),
        axis=1
    )
