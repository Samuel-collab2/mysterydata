import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from core.loader import load_train_dataset
from core.preprocessing import separate_features_label, create_augmented_features, \
    expand_dataset_deterministic, get_categorical_columns
from core.constants_feature_set import SIGNIFICANT_RIDGE_COLUMNS, \
    SIGNIFICANT_AUGMENTED_REGRESSION_COLUMNS, SIGNIFICANT_AUGMENTED_POSITIVE_REGRESSION_COLUMNS
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO



def evaluate_model(model,
                   train_features, train_labels,
                   test_features, test_labels,
                   feature_subset=None,
                   standardize=False):
    if feature_subset is not None:
        train_features = train_features[feature_subset]
        test_features = test_features[feature_subset]

    if standardize:
        scaler = StandardScaler()
        train_features = pd.DataFrame(scaler.fit_transform(train_features),
            columns=train_features.columns)
        test_features = pd.DataFrame(scaler.fit_transform(test_features),
            columns=test_features.columns)

    model = (model()
        if isinstance(model, type)
        else model)
    model.fit(train_features, train_labels)
    print_model_metrics(model, test_features, test_labels)

def print_model_metrics(model, test_features, test_labels):
    error = mean_absolute_error(test_labels, model.predict(test_features))
    print(f'MAE: {error:.4f}')

def sandbox_regression(dataset):
    print('Running regression sandbox...')

    accept_data = dataset[dataset[DATASET_LABEL_NAME] > 0]
    accept_features, accept_labels = separate_features_label(accept_data, DATASET_LABEL_NAME)
    categorical_columns = get_categorical_columns(accept_features)
    determining_data = load_train_dataset()
    determining_features = determining_data.drop(DATASET_LABEL_NAME, axis='columns')
    accept_features_expanded = expand_dataset_deterministic(
        accept_features,
        determining_features,
        categorical_columns
    )

    original_columns = accept_features.columns
    accept_features = pd.concat((
        accept_features,
        accept_features_expanded,
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

    print('\n-- Evaluating standardized linear regression...')
    evaluate_model(LinearRegression, *dataset,
        feature_subset=original_columns,
        standardize=True)

    print('\n-- Evaluating linear regression with ridge features...')
    evaluate_model(LinearRegression, *dataset,
        feature_subset=SIGNIFICANT_RIDGE_COLUMNS)

    print('\n-- Evaluating linear regression using augmented regression features...')
    evaluate_model(LinearRegression, *dataset,
        feature_subset=SIGNIFICANT_AUGMENTED_POSITIVE_REGRESSION_COLUMNS)

    print('\n-- Evaluating standardized linear regression using augmented regression features...')
    evaluate_model(LinearRegression, *dataset,
        feature_subset=SIGNIFICANT_AUGMENTED_POSITIVE_REGRESSION_COLUMNS,
        standardize=True)

    print('\n-- Evaluating 2-degree polynomial regression with ridge features...')
    evaluate_model(make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    ), *dataset, feature_subset=SIGNIFICANT_RIDGE_COLUMNS)

    print('\n-- Evaluating linear regression with ridge features and isolation forest outlier extraction...')
    iso = IsolationForest(n_estimators=100)
    iso.fit(train_features, train_labels)
    pred_inliers = iso.predict(train_features)
    inlier_indices = [i for i, v in enumerate(pred_inliers) if v == 1]
    print(f'Extracted {len(train_features) - len(inlier_indices)} outlier(s)')
    evaluate_model(LinearRegression(),
        train_features.iloc[inlier_indices], train_labels.iloc[inlier_indices],
        test_features, test_labels,
        feature_subset=SIGNIFICANT_RIDGE_COLUMNS)

    print('\n-- Evaluating linear regression with augmented features and 2-component PCA...')
    pca = PCA(n_components=2, random_state=0)
    evaluate_model(LinearRegression(),
        pca.fit_transform(train_features[SIGNIFICANT_AUGMENTED_POSITIVE_REGRESSION_COLUMNS]), train_labels,
        pca.fit_transform(test_features[SIGNIFICANT_AUGMENTED_POSITIVE_REGRESSION_COLUMNS]), test_labels)


if __name__ == '__main__':
    sandbox_regression(dataset=load_train_dataset())
