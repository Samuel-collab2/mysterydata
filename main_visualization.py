from os.path import join

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from core.preprocessing import separate_features_label, create_augmented_features
from core.constants import OUTPUT_DIR, DATASET_LABEL_NAME
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS


def plot_heatmap(features, num_bins, fig_name):
    fig, ax = plt.subplots(1, 2)

    pca_features = pd.DataFrame(features)
    heatmap_features = pca_features.groupby([
        pd.cut(pca_features[1], num_bins),
        pd.cut(pca_features[0], num_bins),
    ]).mean().unstack()
    heatmap_min = heatmap_features.min().min()
    heatmap_max = heatmap_features.max().max()

    def subplot_heatmap(features, ax):
        sns.heatmap(features,
            ax=ax,
            vmin=heatmap_min,
            vmax=heatmap_max,
            cmap='Blues',
            square=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.invert_yaxis()

    subplot_heatmap(heatmap_features[0], ax[0])
    ax[0].set_title('PCA for augmented features for rejected claims')

    subplot_heatmap(heatmap_features[1], ax[1])
    ax[1].set_title('PCA for augmented features for accepted claims')

    fig_path = join(OUTPUT_DIR, fig_name)
    fig.set_figwidth(16)
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Wrote plot to {fig_path}')


def main(dataset):
    features, _ = separate_features_label(dataset, DATASET_LABEL_NAME)
    features_augmented = create_augmented_features(features, SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS)

    pca = PCA(n_components=2, random_state=0)
    pca_points = pca.fit_transform(features_augmented)
    plot_heatmap(pca_points,
        num_bins=7,
        fig_name='heatmap_augmented_pca.png')


if __name__ == '__main__':
    from core.loader import load_train_dataset
    main(dataset=load_train_dataset())
