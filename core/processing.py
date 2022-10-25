import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import MIN_REAL_FEATURE_UNIQUE_VALUES

def split_dataset_features_label(dataset, label_name):
    return (
        dataset.drop(label_name, axis='columns', inplace=False),
        dataset.loc[:, label_name],
    )

def split_data_training_test(data, train_factor):
    return train_test_split(*data, train_size=train_factor, shuffle=False)

def is_categorical_column(column):
    return column.dtype == 'object' or column.nunique() < MIN_REAL_FEATURE_UNIQUE_VALUES

def expand_dataset(dataset_raw):
    """
    Expands the dataset by splitting each categorical column into many
    :param dataset_raw: the raw dataset
    :return: the expanded dataset
    """

    categorical_columns = [column for column in dataset_raw.columns
                           if is_categorical_column(dataset_raw[column])]

    data_expanded = pd.get_dummies(dataset_raw,
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

    data_columns = list(dataset_raw.columns)
    return data_expanded.reindex(
        sorted(data_expanded.columns,
        key=index_column),
        axis=1
    )
