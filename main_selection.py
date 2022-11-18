from os.path import join
from itertools import product, combinations
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import r_regression

from core.preprocessing import separate_features_label, convert_label_boolean
from core.constants import DATASET_LABEL_NAME, OUTPUT_DIR


def _augment_features(features):
    print(f'Augmenting {len(features.columns)}-feature dataset...')
    features_augmented = pd.DataFrame()
    feature_combinations = combinations(features, r=2)

    for feature1, feature2 in feature_combinations:
        feature1_data = features[feature1]
        feature2_data = features[feature2]

        features_product = pd.Series(feature1_data * feature2_data,
            name=f'{feature1}*{feature2}')

        features_quotient = pd.Series(feature1_data / feature2_data,
            name=f'{feature1}/{feature2}')

        features_augmented = pd.concat((
            features_augmented,
            features_product,
            features_quotient,
        ), axis='columns')

    return features_augmented


def print_correlations(correlations_matrix, significance_threshold=0, column_names=None):
    column_names = column_names or correlations_matrix.columns
    print(f'\n-- Printing correlations with significant feature threshold {significance_threshold}...')

    def order_correlation(correlation):
        (_, column_values), row_name = correlation
        return abs(column_values[row_name])

    correlations_visited = set()
    correlations = sorted(
        product(correlations_matrix.iteritems(), column_names),
        key=order_correlation,
        reverse=True,
    )
    for (column_name, column_values), row_name in correlations:
        if (column_name == row_name
        or (row_name, column_name) in correlations_visited):
            continue

        correlations_visited.add((column_name, row_name))

        combination_significance = column_values[row_name]
        if abs(combination_significance) < significance_threshold:
            continue

        print(f'{column_name} x {row_name} -> {combination_significance:.4f}')


def main(dataset, boolean=False, standardize=False):
    dataset.drop('rowIndex', axis='columns', inplace=True)

    if standardize:
        dataset = dataset[dataset[DATASET_LABEL_NAME] > 0]

    features, labels = separate_features_label(dataset, DATASET_LABEL_NAME)

    if standardize:
        features = pd.DataFrame(
            StandardScaler().fit_transform(features),
            columns=features.columns
        )

    if boolean:
        labels = convert_label_boolean(labels)

    correlations_path = join(OUTPUT_DIR, 'correlations.csv')
    pd.concat((
        pd.Series(features.columns, name='feature'),
        pd.Series(r_regression(features, labels), name=DATASET_LABEL_NAME),
    ), axis='columns').to_csv(correlations_path, index=False)
    print(f'Wrote correlations matrix to {correlations_path}')

    NUM_ITERS = 5
    FEATURE_SET_SIZE = 15
    SIGNIFICANCE_THRESHOLD = 0.1
    for i in range(NUM_ITERS):
        print(f'\n-- Iteration {i+1}')

        features_augmented = pd.concat((
            features,
            _augment_features(features)
        ), axis='columns')
        features_augmented.fillna(0, inplace=True)
        features_augmented.replace(float('inf'), 0, inplace=True)

        feature_correlations = sorted(zip(
            features_augmented.columns,
            r_regression(features_augmented, labels),
        ), key=lambda v: v[1], reverse=True)
        feature_correlations = {f: c
            for f, c in feature_correlations}

        best_correlations = list(feature_correlations.items())[:FEATURE_SET_SIZE]
        best_correlation_names, _ = zip(*best_correlations)
        print('\n'.join((f'{f} -> {c:.4f}'
            for f, c in best_correlations)))
        features = features_augmented.loc[:, best_correlation_names].copy()
        features = features.loc[:, ~features.columns.duplicated()]

if __name__ == '__main__':
    from core.loader import load_train_dataset
    main(dataset=load_train_dataset(), boolean=True)
