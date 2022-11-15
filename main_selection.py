from os.path import join
import pandas as pd
from itertools import product

from core.preprocessing import separate_features_label, convert_label_boolean
from core.constants import DATASET_LABEL_NAME, OUTPUT_DIR


def main(dataset, correlation_threshold=0.2):
    features, labels = separate_features_label(dataset, DATASET_LABEL_NAME)
    labels_boolean = convert_label_boolean(labels)

    dataset_processed = pd.concat((
        features.drop('rowIndex', axis='columns', inplace=False),
        labels_boolean
    ), axis='columns')

    correlations_matrix = dataset_processed.corr()
    correlations_path = join(OUTPUT_DIR, 'correlations.csv')
    correlations_matrix.to_csv(correlations_path)
    print(f'Wrote correlations matrix to {correlations_path}')

    print(f'Printing correlations with significant feature threshold {correlation_threshold}')
    correlations = product(correlations_matrix.iteritems(), correlations_matrix.columns)
    correlations_visited = set()
    for (column_name, column_values), row_name in correlations:
        if (column_name == row_name
        or (row_name, column_name) in correlations_visited):
            continue

        correlations_visited.add((column_name, row_name))

        combination_significance = column_values[row_name]
        if abs(combination_significance) < correlation_threshold:
            continue

        print(f'{column_name} x {row_name} -> {combination_significance:.4f}')


if __name__ == '__main__':
    from core.loader import load_train_dataset
    main(dataset=load_train_dataset())
