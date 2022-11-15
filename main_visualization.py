from os.path import join

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from core.preprocessing import separate_features_label, create_augmented_features, \
    convert_label_boolean
from core.constants import OUTPUT_DIR, DATASET_LABEL_NAME, \
    SIGNIFICANT_AUGMENTED_COLUMNS


def main(dataset):
    features, labels = separate_features_label(dataset, DATASET_LABEL_NAME)
    features_augmented = create_augmented_features(features, SIGNIFICANT_AUGMENTED_COLUMNS)
    labels_boolean = convert_label_boolean(labels)

    pca = PCA(n_components=2, random_state=0)
    features_pca = pca.fit_transform(features_augmented)

    label_points = {label: [
        features_pca[i] for i, l in enumerate(labels_boolean)
            if l == label
    ] for label in labels_boolean.unique()}

    fig, ax = plt.subplots()
    ax.scatter(*zip(*label_points[False]), alpha=1/2, label='Rejected claim')
    ax.scatter(*zip(*label_points[True]), alpha=1/2, label='Accepted claim')
    ax.set_title('Principal component analysis for augmented feature set')
    ax.legend()

    fig_path = join(OUTPUT_DIR, 'scatter_augmented_pca.png')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Wrote plot to {fig_path}')


if __name__ == '__main__':
    from core.loader import load_train_dataset
    main(dataset=load_train_dataset())
