import math
from collections import OrderedDict
from itertools import combinations

import numpy as np
from sklearn.metrics import mean_absolute_error

from core.constants import ANALYSIS_LASSO_LAMBDAS, ANALYSIS_RIDGE_LAMBDAS, ANALYSIS_CROSS_VALIDATION_SETS, \
    ANALYSIS_POLYNOMIAL_DEGREES, ANALYSIS_SIGNIFICANT_FEATURE_COUNT, ANALYSIS_CORRELATION_THRESHOLD
from core.preprocessing import enumerate_cross_validation_sets, zip_sort
from core.model_regression import train_linear_regression, train_polynomial_regression, train_lasso_regression, train_ridge_regression

def cross_validate_model(train_model, features, label, sets):
    train_errors = []
    cv_errors = []
    for cross_validation_set in enumerate_cross_validation_sets(features, label, sets):
        cv_train_features, cv_train_label, cv_valid_features, cv_valid_label = cross_validation_set

        model = train_model(cv_train_features, cv_train_label)
        train_predictions = model.predict(cv_train_features)
        valid_predictions = model.predict(cv_valid_features)

        train_errors.append(mean_absolute_error(cv_train_label, train_predictions))
        cv_errors.append(mean_absolute_error(cv_valid_label, valid_predictions))

    return np.average(train_errors), np.average(cv_errors)

def _enumerate_cross_validation_errors(train_model, train_features, train_label, cv_sets, domain):
    for iteration in domain:
        yield iteration, *cross_validate_model(
            lambda features, label: train_model(train_features, train_label, iteration),
            train_features,
            train_label,
            cv_sets,
        )

def calculate_linear_regression_error(train_features, train_label, test_features, test_label):
    model = train_linear_regression(train_features, train_label)
    predictions = model.predict(test_features)
    return mean_absolute_error(test_label, predictions)

def calculate_cross_validation_errors(train_model, train_features, train_label, cv_sets, domain):
    return _enumerate_cross_validation_errors(
        lambda features, label, iteration: train_model(features, label, iteration),
        train_features,
        train_label,
        cv_sets,
        domain,
    )

def _perform_cross_validation_analysis(train_model, train_data, domain, domain_name):
    cv_sets = ANALYSIS_CROSS_VALIDATION_SETS
    print(f'Calculating error across {domain_name}s: {domain}')
    print(f'Using {cv_sets} cross-validation sets')

    min_cv_error = math.inf
    min_cv_iteration = None
    for iteration, train_error, cv_error in calculate_cross_validation_errors(train_model, *train_data, cv_sets, domain):
        print(f'{domain_name}={iteration}')
        print(f'|_ Train error: {train_error:.4f}')
        print(f'|_ Validation error: {cv_error:.4f}')

        if cv_error < min_cv_error:
            min_cv_error = cv_error
            min_cv_iteration = iteration

    print(f'Lowest validation error: {min_cv_error:.4f}')
    print(f'Lowest validation error {domain_name}: {min_cv_iteration:.4f}')

    return min_cv_iteration

def _perform_composite_analysis(train_model, train_data, test_data, domain, domain_name):
    min_cv_iteration = _perform_cross_validation_analysis(train_model, train_data, domain, domain_name)
    print(f'Model fit at {domain_name}={min_cv_iteration}')
    model = train_model(*train_data, min_cv_iteration)

    test_features, test_label = test_data
    predictions = model.predict(test_features)

    mae = mean_absolute_error(test_label, predictions)
    print(f'Model MAE: {mae:.4f}')

    return model


def perform_linear_regression_analysis(train_data, test_data):
    print('Performing linear regression analysis...')
    mae = calculate_linear_regression_error(*train_data, *test_data)
    print(f'Model MAE: {mae:.4f}')

def perform_polynomial_complexity_analysis(train_data, test_data):
    print('Performing polynomial complexity analysis...')
    _perform_composite_analysis(
        train_polynomial_regression,
        train_data,
        test_data,
        range(1, ANALYSIS_POLYNOMIAL_DEGREES + 1),
        "degree"
    )

def _perform_lambda_analysis(train_model, train_data, test_data, lambdas):
    train_features, _ = train_data
    model = _perform_composite_analysis(
        train_model,
        train_data,
        test_data,
        lambdas,
        "lambda"
    )

    coefficients, columns = zip_sort(model.coef_, train_features.columns, comparator=lambda x: abs(x[0]), reverse=True)

    print(f'{ANALYSIS_SIGNIFICANT_FEATURE_COUNT} Most significant columns at lambda={model.alpha}')
    for i in range(ANALYSIS_SIGNIFICANT_FEATURE_COUNT + 1):
        print(f"{columns[i]}: {coefficients[i]:.4f}")

def perform_lasso_lambda_analysis(train_data, test_data):
    print('Performing lasso lambda analysis...')
    _perform_lambda_analysis(
        train_lasso_regression,
        train_data,
        test_data,
        ANALYSIS_LASSO_LAMBDAS,
    )

def perform_ridge_lambda_analysis(train_data, test_data):
    print('Performing ridge lambda analysis...')
    _perform_lambda_analysis(
        train_ridge_regression,
        train_data,
        test_data,
        ANALYSIS_RIDGE_LAMBDAS,
    )

def perform_feature_correlation_analysis(features):
    print('Performing feature correlation analysis...')
    print(f'Correlation threshold > {ANALYSIS_CORRELATION_THRESHOLD}')
    for column1, column2 in combinations(features.columns, r=2):
        feature1 = features.loc[:, column1]
        feature2 = features.loc[:, column2]

        correlation = feature1.corr(feature2)
        if abs(correlation) > ANALYSIS_CORRELATION_THRESHOLD:
            print(f'{column1} x {column2}: {correlation}')
