import numpy as np
from sklearn.metrics import mean_absolute_error

from core.constants import ANALYSIS_LASSO_LAMBDAS, CROSS_VALIDATION_SETS, ANALYSIS_POLYNOMIAL_DEGREES
from core.processing import enumerate_cross_validation_sets
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

def _enumerate_cross_validation_errors(train_model, train_features, train_label, cv_sets, iterable):
    for item in iterable:
        yield item, *cross_validate_model(
            lambda features, label: train_model(train_features, train_label, item),
            train_features,
            train_label,
            cv_sets,
        )

def calculate_linear_regression_error(train_features, train_label, test_features, test_label):
    model = train_linear_regression(train_features, train_label)
    predictions = model.predict(test_features)
    return mean_absolute_error(test_label, predictions)

def calculate_polynomial_complexity_errors(train_features, train_label, cv_sets, degrees):
    return _enumerate_cross_validation_errors(
        lambda features, label, degree: train_polynomial_regression(features, label, degree),
        train_features,
        train_label,
        cv_sets,
        degrees,
    )

def calculate_lasso_lambda_errors(train_features, train_label, cv_sets, lambdas):
    return _enumerate_cross_validation_errors(
        lambda features, label, alpha: train_lasso_regression(features, label, alpha),
        train_features,
        train_label,
        cv_sets,
        lambdas,
    )

def calculate_ridge_lambda_errors(train_features, train_label, cv_sets, lambdas):
    return _enumerate_cross_validation_errors(
        lambda features, label, alpha: train_ridge_regression(features, label, alpha),
        train_features,
        train_label,
        cv_sets,
        lambdas,
    )

def perform_linear_regression_analysis(train_data, test_data):
    print('Calculating linear regression mae...')
    linear_regression_mae = calculate_linear_regression_error(*train_data, *test_data)
    print(f'Linear regression MAE: {linear_regression_mae:.4f}')

def perform_polynomial_complexity_analysis(train_data, test_data):
    degrees = range(1, ANALYSIS_POLYNOMIAL_DEGREES + 1)
    cv_sets = CROSS_VALIDATION_SETS
    print(f'Calculating polynomial regression mae for degrees={degrees} across {cv_sets} cross-validation sets...')
    for degree, train_error, cv_error in calculate_polynomial_complexity_errors(*train_data, degrees=degrees, cv_sets=cv_sets):
        print(f'Polynomial regression degree={degree}')
        print(f'|_ Train error: {train_error:.4f}')
        print(f'|_ Validation error: {cv_error:.4f}')

def perform_lasso_lambda_analysis(train_data, test_data):
    lambdas = ANALYSIS_LASSO_LAMBDAS
    cv_sets = CROSS_VALIDATION_SETS
    print(f'Calculating lasso regression mae for lambdas={lambdas} across {cv_sets} cross-validation sets...')
    for alpha, train_error, cv_error in calculate_lasso_lambda_errors(*train_data, lambdas=lambdas, cv_sets=cv_sets):
        print(f'Lasso regression λ={alpha}')
        print(f'|_ Train error: {train_error:.4f}')
        print(f'|_ Validation error: {cv_error:.4f}')

def perform_ridge_lambda_analysis(train_data, test_data):
    lambdas = ANALYSIS_LASSO_LAMBDAS
    cv_sets = CROSS_VALIDATION_SETS
    print(f'Calculating ridge regression mae for lambdas={lambdas} across {cv_sets} cross-validation sets...')
    for alpha, train_error, cv_error in calculate_ridge_lambda_errors(*train_data, lambdas=lambdas, cv_sets=cv_sets):
        print(f'Lasso regression λ={alpha}')
        print(f'|_ Train error: {train_error:.4f}')
        print(f'|_ Validation error: {cv_error:.4f}')
