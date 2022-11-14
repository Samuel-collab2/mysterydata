import pandas as pd
import core.preprocessing as proc
import core.model_regression as reg
from sklearn.metrics import mean_absolute_error
import core.data_analysis as da
from sklearn.model_selection import KFold

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from core.constants import ANALYSIS_LASSO_LAMBDAS, ANALYSIS_RIDGE_LAMBDAS, ANALYSIS_CROSS_VALIDATION_SETS, \
    ANALYSIS_POLYNOMIAL_DEGREES, ANALYSIS_SIGNIFICANT_FEATURE_COUNT, ANALYSIS_CORRELATION_THRESHOLD
import math

import sys


def get_my_data_bb(train_factor=0.75, subset_size=100):
    # load data
    data = pd.read_csv('../trainingset.csv', low_memory=False)

    # # use subset of dataset
    # data = data.sample(subset_size)

    # get categorical columns
    categorical_cols = proc.get_categorical_columns(data)

    # apply one hot encoding to all categorical values
    expanded_dataset = proc.expand_dataset(data, categorical_cols)

    # keep only selected features
    # expanded_dataset = expanded_dataset.iloc[['feature1', 'feature17', 'ClaimAmount']]
    expanded_dataset = expanded_dataset.loc[:, ['feature1', 'feature15_4', 'feature3_8', 'feature17', 'feature15_6', 'feature13_2', 'feature15_1', 'feature18_1', 'feature7_2', 'feature7_3', 'feature9_0', 'feature15_8', 'feature15_2', 'feature11_6', 'feature3_4', 'feature3_6', 'feature10', 'feature13_3', 'feature9_1', 'feature13_0', 'ClaimAmount']]

    # DataFrame, and series.Series << feature and label types
    # split_data = proc.separate_features_label_and_remove_rowIndex_col(expanded_dataset, 'ClaimAmount')
    split_data = proc.separate_features_label(expanded_dataset, 'ClaimAmount')

    # split samples by claim_amount > 0 <split_data_by_claim[0]>, and ClaimAmount == 1 <split_data_by_claim[1]>
    split_data_by_claim = proc.split_claims_accept_reject(split_data[0], split_data[1])
    split_data_train_test = proc.split_training_test(split_data_by_claim[0][0], split_data_by_claim[0][1], train_factor)

    # print(split_data_train_test[0][1])
    # print(split_data_train_test[1][1])
    return split_data_train_test


def get_size_of_data():
    # load data
    data = pd.read_csv('../trainingset.csv', low_memory=False)
    return data.shape[0]

def temp_data_load():
    # load data
    data = pd.read_csv('../trainingset.csv', low_memory=False)
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

    # split samples by claim_amount > 0 <split_data_by_claim[0]>, and ClaimAmount == 1 <split_data_by_claim[1]>
    split_data_by_claim = proc.split_claims_accept_reject(expanded_dataset[0], expanded_dataset[1])
    split_data_train_test = proc.split_training_test(split_data_by_claim[0][0], split_data_by_claim[0][1], 0.75)
    return split_data_train_test


def learning_curve_script():
    # # [0] = training.   [0][0] = features.  [0][1] = labels
    # # [1] = test.       [1][0] = ...
    split_data_train_test = get_my_data_bb(0.75)
    training_features = split_data_train_test[0][0]
    training_labels = split_data_train_test[0][1]
    test_features = split_data_train_test[1][0]
    test_labels = split_data_train_test[1][1]

    cv_errors = []
    training_errors = []

    kf = KFold(n_splits=2)
    kf.get_n_splits(training_features)

    for train_index, test_index in kf.split(training_features):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = training_features[train_index], training_features[test_index]
        y_train, y_test = training_labels[train_index], training_labels[test_index]
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

    # train_features_subset = training_features.iloc[0:current_chunk_size]
    # train_labels_subset = training_labels.iloc[0:current_chunk_size]
    # for i in range(0, 10):
    #     train_error, cv_error = _perform_cross_validation_analysis(
    #         reg.train_polynomial_regression,
    #         (train_features_subset, train_labels_subset),
    #         range(2, 3),
    #         'degree'
    #     )
    #     cv_errors.append(cv_error)
    #     training_errors.append(train_error)
    #     current_chunk_size += incremental_chunk_size
    #     print(current_chunk_size)
    #     train_features_subset = training_features.iloc[0:current_chunk_size]
    #     train_labels_subset = training_labels.iloc[0:current_chunk_size]

    # print(training_errors)
    # print(cv_errors)

    # model = reg.train_polynomial_regression(training_features, training_labels, 3)
    # pred = model.predict(test_features)
    # mae = mean_absolute_error(test_labels, pred)
    # print(mae)


# def learning_curve_script():
#     # # [0] = training.   [0][0] = features.  [0][1] = labels
#     # # [1] = test.       [1][0] = ...
#     split_data_train_test = get_my_data_bb(0.75)
#     training_features = split_data_train_test[0][0]
#     training_labels = split_data_train_test[0][1]
#     test_features = split_data_train_test[1][0]
#     test_labels = split_data_train_test[1][1]
#
#     size_data = get_size_of_data()
#     incremental_chunk_size = int(size_data / 10)
#     current_chunk_size = incremental_chunk_size
#     cv_errors = []
#     training_errors = []
#
#     train_features_subset = training_features.iloc[0:current_chunk_size]
#     train_labels_subset = training_labels.iloc[0:current_chunk_size]
#     for i in range(0, 10):
#         train_error, cv_error = _perform_cross_validation_analysis(
#             reg.train_polynomial_regression,
#             (train_features_subset, train_labels_subset),
#             range(2, 3),
#             'degree'
#         )
#         cv_errors.append(cv_error)
#         training_errors.append(train_error)
#         current_chunk_size += incremental_chunk_size
#         print(current_chunk_size)
#         train_features_subset = training_features.iloc[0:current_chunk_size]
#         train_labels_subset = training_labels.iloc[0:current_chunk_size]
#
#     # print(training_errors)
#     # print(cv_errors)
#
#     # model = reg.train_polynomial_regression(training_features, training_labels, 3)
#     # pred = model.predict(test_features)
#     # mae = mean_absolute_error(test_labels, pred)
#     # print(mae)


def _perform_cross_validation_analysis(train_model, train_data, domain, domain_name):
    cv_sets = ANALYSIS_CROSS_VALIDATION_SETS
    print(f'Calculating error across {domain_name}s: {domain}')
    print(f'Using {cv_sets} cross-validation sets')

    # iteration, train_error, cv_error = da.calculate_cross_validation_errors(train_model, *train_data, cv_sets, domain)
    for iteration, train_error, cv_error in da.calculate_cross_validation_errors(train_model, *train_data, cv_sets, domain):
        print(train_error)
        print(cv_error)
        return train_error, cv_error


# learning_curve_script()
print(temp_data_load())
