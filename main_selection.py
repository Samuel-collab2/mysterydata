from os.path import join
import pandas as pd

from core.preprocessing import separate_features_label, convert_label_boolean
from core.constants import DATASET_LABEL_NAME, OUTPUT_DIR


def main(dataset):
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


if __name__ == '__main__':
    from core.loader import load_train_dataset
    main(dataset=load_train_dataset())
