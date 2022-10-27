import numpy as np
from sklearn.metrics import mean_absolute_error

from core.preprocessing import enumerate_cross_validation_sets
from core.model_regression import train_linear_regression, train_polynomial_regression

def cross_validate_model(train_model, features, label, sets):
    train_errors = []
    cv_errors = []

    for cross_validation_set in enumerate_cross_validation_sets(features, label, sets):
        cv_train_features, cv_train_label, cv_valid_features, cv_valid_label = cross_validation_set

        model = train_model(cv_train_features, cv_train_label)
        training_predictions = model.predict(cv_train_features)
        validation_predictions = model.predict(cv_valid_features)

        train_errors.append(mean_absolute_error(cv_train_label, training_predictions))
        cv_errors.append(mean_absolute_error(cv_valid_label, validation_predictions))

    return np.average(train_errors), np.average(cv_errors)

def calculate_linear_regression_error(train_features, train_label, test_features, test_label):
    model = train_linear_regression(train_features, train_label)
    predictions = model.predict(test_features)
    return mean_absolute_error(test_label, predictions)

def calculate_polynomial_complexity_errors(train_features, train_label, degrees, cv_sets):
    for degree in degrees:
        yield degree, *cross_validate_model(
            lambda features, label: train_polynomial_regression(features, label, degree=degree),
            train_features,
            train_label,
            cv_sets,
        )
