"""
COMP 4983 - Lab 6
Standardization, ridge regression, and lasso (loader component)
Brandon Semilla (A01233758, Set M)
"""

import pandas as pd


def load(filepath):
    """
    Loads the CSV dataset for this lab.
    :param filepath: The file path to load the dataset from.
    :return: a pandas DataFrame
    """

    data = pd.read_csv(filepath, low_memory=False)

    categorical_columns = [c for c in data.columns
        if data.dtypes[c] == 'object'
        or data[c].nunique() < 20]

    data_expanded = pd.get_dummies(data,
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


    data_columns = list(data.columns)
    return data_expanded.reindex(sorted(data_expanded.columns,
        key=index_column), axis=1)
