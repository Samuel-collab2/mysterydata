import pandas as pd
import core.preprocessing as proc
import core.model_regression as reg
from sklearn.metrics import mean_absolute_error
import core.data_analysis as da
from sklearn.model_selection import KFold
import random
from core.loader import load_train_dataset, load_standardized_train_dataset, load_determining_dataset

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, MENU_EXIT, MENU_RETURN, SIGNIFICANT_RIDGE_COLUMNS, \
    SIGNIFICANT_BINARY_LABEL_COLUMNS, SIGNIFICANT_FORWARD_STEPWISE_COLUMNS, SIGNIFICANT_FEATURE_SET_COUNTS, \
    ANALYSIS_CROSS_VALIDATION_SETS, ANALYSIS_POLYNOMIAL_DEGREES
import math

import sys

# def get_my_data_bb(train_factor=0.75, subset_size=100):
#     # load data
#     data = pd.read_csv('../trainingset.csv', low_memory=False, delim_whitespace=True)
#
#     # # use subset of dataset
#     # data = data.sample(subset_size)
#
#     # get categorical columns
#     categorical_cols = proc.get_categorical_columns(data)
#
#     # apply one hot encoding to all categorical values
#     expanded_dataset = proc.expand_dataset(data, categorical_cols)
#
#     # keep only selected features
#     # expanded_dataset = expanded_dataset.iloc[['feature1', 'feature17', 'ClaimAmount']]
#     expanded_dataset = expanded_dataset.loc[:, ['feature1', 'feature15_4', 'feature3_8', 'feature17', 'feature15_6', 'feature13_2', 'feature15_1', 'feature18_1', 'feature7_2', 'feature7_3', 'feature9_0', 'feature15_8', 'feature15_2', 'feature11_6', 'feature3_4', 'feature3_6', 'feature10', 'feature13_3', 'feature9_1', 'feature13_0', 'ClaimAmount']]
#
#     # DataFrame, and series.Series << feature and label types
#     # split_data = proc.separate_features_label_and_remove_rowIndex_col(expanded_dataset, 'ClaimAmount')
#     split_data = proc.separate_features_label(expanded_dataset, 'ClaimAmount')
#
#     # split samples by claim_amount > 0 <split_data_by_claim[0]>, and ClaimAmount == 1 <split_data_by_claim[1]>
#     split_data_by_claim = proc.split_claims_accept_reject(split_data[0], split_data[1])
#     split_data_train_test = proc.split_training_test(split_data_by_claim[0][0], split_data_by_claim[0][1], train_factor)
#
#     # print(split_data_train_test[0][1])
#     # print(split_data_train_test[1][1])
#     return split_data_train_test

def temp_data_load(subset_fraction=1.0):
    # load data
    data = pd.read_csv('../trainingset.csv', low_memory=False)

    # use only specified fraction of data (shuffled)
    data = data.sample(frac=subset_fraction)

    # get categorical columns
    categorical_cols = proc.get_categorical_columns(data)

    # apply one hot encoding to all categorical values
    expanded_dataset = proc.expand_dataset(data, categorical_cols)

    # keep only selected features
    expanded_dataset = expanded_dataset.loc[:,
                       ['feature1', 'feature15_4', 'feature3_8', 'feature17', 'feature15_6', 'feature13_2',
                        'feature15_1', 'feature18_1', 'feature7_2', 'feature7_3', 'feature9_0', 'feature15_8',
                        'feature15_2', 'feature11_6', 'feature3_4', 'feature3_6', 'feature10', 'feature13_3',
                        'feature9_1', 'feature13_0', 'ClaimAmount']]

    expanded_dataset = expanded_dataset[expanded_dataset["ClaimAmount"] > 0]

    # split the data into training set and test set
    # use 75 percent of the data to train the model and hold back 25 percent
    # for testing
    train_ratio = 0.75
    # number of samples in the data_subset
    num_rows = expanded_dataset.shape[0]
    # shuffle the indices
    shuffled_indices = list(range(num_rows))
    random.seed(42)
    random.shuffle(shuffled_indices)

    # calculate the number of rows for training
    train_set_size = int(num_rows * train_ratio)
    # print("total num rows:", num_rows)
    # print("training set: ", train_set_size)

    # training set: take the first 'train_set_size' rows
    train_indices = shuffled_indices[:train_set_size]
    # test set: take the remaining rows
    test_indices = shuffled_indices[train_set_size:]

    # create training set and test set
    train_data = expanded_dataset.iloc[train_indices, :]
    test_data = expanded_dataset.iloc[test_indices, :]
    # print(len(train_data), "training samples + ", len(test_data), "test samples")

    # prepare training features and training labels
    # features: all columns except 'price'
    # labels: 'ClaimAmount' column
    train_features = train_data.drop('ClaimAmount', axis=1, inplace=False)
    train_labels = train_data.loc[:, 'ClaimAmount']

    # prepare test features and test labels
    test_features = test_data.drop('ClaimAmount', axis=1, inplace=False)
    test_labels = test_data.loc[:, 'ClaimAmount']

    return (train_features, train_labels), (test_features, test_labels)


def learning_curve_script():
    split_data_train_test = temp_data_load()
    train_data = split_data_train_test[0][0]
    train_label = split_data_train_test[0][1]
    test_data = split_data_train_test[1][0]
    test_label = split_data_train_test[1][1]

    cv_error_sum = 0
    kf = KFold(n_splits=5)
    kf.get_n_splits(train_data)
    model_complexity = 1
    index = 1

    for train_index, test_index in kf.split(train_data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_test = train_label.iloc[train_index], train_label.iloc[test_index]
        model = reg.train_polynomial_regression(X_train, y_train, model_complexity)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        print("mae for K-fold training set %d: %f " % (index, mae))
        index += 1
        # if mae > 3000:
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #     print(X_train)
        #     print(y_train)
        #     print(X_test)
        #     print(y_test)


def learning_curve_script2():
    dataset_raw = pd.read_csv("../trainingset.csv", low_memory=False)

    raw_features, raw_label = proc.separate_features_label(dataset_raw, DATASET_LABEL_NAME)

    categorical_columns = proc.get_categorical_columns(raw_features)
    dataset_expanded = proc.expand_dataset_deterministic(dataset_raw, dataset_raw, categorical_columns)

    expanded_features, expanded_label = proc.separate_features_label(dataset_expanded, DATASET_LABEL_NAME)

    # binary_label = convert_label_binary(expanded_label)

    (accept_features, accept_label), (reject_features, reject_label) = proc.split_claims_accept_reject(
        expanded_features,
        expanded_label
    )

    features_reduced_dim = accept_features.loc[:,
                       ['feature1', 'feature15_4', 'feature3_8', 'feature17', 'feature15_6', 'feature13_2',
                        'feature15_1', 'feature18_1', 'feature7_2', 'feature7_3', 'feature9_0', 'feature15_8',
                        'feature15_2', 'feature11_6', 'feature3_4', 'feature3_6', 'feature10', 'feature13_3',
                        'feature9_1', 'feature13_0']]

    train_data, test_data = proc.split_training_test(
        features_reduced_dim,
        accept_label,
        DATASET_TRAIN_RATIO
    )

    da._perform_cross_validation_analysis_2(
        reg.train_polynomial_regression,
        train_data,
        range(1, ANALYSIS_POLYNOMIAL_DEGREES + 1),
        'degree'
    )

learning_curve_script2()
# learning_curve_script()
