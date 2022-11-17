from os.path import join
from itertools import product, combinations
import pandas as pd

from sklearn.preprocessing import StandardScaler

from core.preprocessing import separate_features_label, convert_label_boolean
from core.constants import DATASET_LABEL_NAME, OUTPUT_DIR


def _augment_features(features):
    print(f'Augmenting {len(features.columns)}-feature dataset...')
    features_augmented = pd.DataFrame()
    feature_combinations = combinations(features, r=2)

    for feature1, feature2 in feature_combinations:
        feature1_data = features.loc[:, feature1]
        feature2_data = features.loc[:, feature2]

        features_product = pd.DataFrame(feature1_data * feature2_data,
            columns=[f'{feature1}*{feature2}'])
        features_quotient = pd.DataFrame(feature1_data / feature2_data,
            columns=[f'{feature1}/{feature2}'])

        features_augmented = pd.concat((
            features_augmented,
            features_product,
            features_quotient
        ), axis='columns')

    return features_augmented


def print_correlations(correlations_matrix, significance_threshold=0, column_names=None):
    column_names = column_names or correlations_matrix.columns
    print(f'Printing correlations with significant feature threshold {significance_threshold}...')

    def order_correlation(correlation):
        (_, column_values), row_name = correlation
        return column_values[row_name]

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
        features = pd.DataFrame(StandardScaler().fit_transform(features), columns=features.columns)

    if boolean:
        labels = convert_label_boolean(labels)

    dataset_processed = pd.concat((features, labels), axis='columns')

    correlations_matrix = dataset_processed.corr()
    correlations_path = join(OUTPUT_DIR, 'correlations.csv')
    correlations_matrix.to_csv(correlations_path)
    print(f'Wrote correlations matrix to {correlations_path}')

    features_augmented = _augment_features(features)
    dataset_augmented = pd.concat((features, features_augmented, labels), axis='columns')

    # search for correlations between augmented features and claim amount
    print_correlations(dataset_augmented.corr(),
        column_names=(DATASET_LABEL_NAME,),
        significance_threshold=0.1)


if __name__ == '__main__':
    from core.loader import load_train_dataset
    main(dataset=load_train_dataset(), boolean=True)
