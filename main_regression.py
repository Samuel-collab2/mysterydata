import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from core.preprocessing import separate_features_label, create_augmented_features
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_REGRESSION_COLUMNS
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO



def evaluate_model(model_type,
                   train_features, train_labels,
                   test_features, test_labels,
                   feature_subset):
    model = model_type()
    model.fit(train_features[feature_subset], train_labels)
    print_model_metrics(model, test_features[feature_subset], test_labels)

def print_model_metrics(model, test_features, test_labels):
    error = mean_absolute_error(test_labels, model.predict(test_features))
    print(f'Mean absolute error: {error:.4f}')

def sandbox_regression(dataset):
    print('Running regression sandbox...')

    accept_data = dataset[dataset[DATASET_LABEL_NAME] > 0]
    accept_features, accept_labels = separate_features_label(accept_data, DATASET_LABEL_NAME)
    original_columns = accept_features.columns
    accept_features = pd.concat((
        accept_features,
        create_augmented_features(accept_features, SIGNIFICANT_AUGMENTED_REGRESSION_COLUMNS)
    ), axis='columns')
    train_features, test_features, train_labels, test_labels = train_test_split(
        accept_features,
        accept_labels,
        train_size=DATASET_TRAIN_RATIO,
    )
    dataset = train_features, train_labels, test_features, test_labels

    print('\n-- Evaluating linear regression...')
    evaluate_model(LinearRegression, *dataset,
        feature_subset=original_columns)

    print('\n-- Evaluating linear regression using augmented regression columns...')
    evaluate_model(LinearRegression, *dataset,
        feature_subset=SIGNIFICANT_AUGMENTED_REGRESSION_COLUMNS)


if __name__ == '__main__':
    from core.loader import load_train_dataset
    sandbox_regression(dataset=load_train_dataset())
