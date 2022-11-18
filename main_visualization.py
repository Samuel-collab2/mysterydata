from os.path import join

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from core.preprocessing import separate_features_label, create_augmented_features, \
    convert_label_boolean
from core.constants import OUTPUT_DIR, DATASET_LABEL_NAME
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS


def main(dataset):
    features, labels = separate_features_label(dataset, DATASET_LABEL_NAME)
    features_augmented = create_augmented_features(features, SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS)
    labels_boolean = convert_label_boolean(labels)

    pca = PCA(n_components=2, random_state=0)
    pca_points = pca.fit_transform(features_augmented)

    fig, ax = plt.subplots()

    NUM_BINS = 7
    pca_features = pd.DataFrame(pca_points)
    heatmap_features = pca_features.groupby([
        pd.cut(pca_features[1], NUM_BINS),
        pd.cut(pca_features[0], NUM_BINS),
    ]).mean().unstack()

    sns.heatmap(heatmap_features,
        ax=ax,
        cmap='Blues',
        square=True)
    ax.set_title('PCA for augmented features')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.invert_yaxis()

    fig_path = join(OUTPUT_DIR, 'heatmap_augmented_pca.png')
    fig.set_figwidth(16)
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Wrote plot to {fig_path}')


if __name__ == '__main__':
    from core.loader import load_train_dataset
    main(dataset=load_train_dataset())
