from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from core.model_regression import train_linear_regression
from core.preprocessing import separate_features_label
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_REGRESSION_COLUMNS
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO


def sandbox_regression(dataset):
    print('Running regression sandbox...')

    accept_data = dataset[dataset[DATASET_LABEL_NAME] > 0]
    accept_features, accept_labels = separate_features_label(accept_data, DATASET_LABEL_NAME)
    train_features, test_features, train_labels, test_labels = train_test_split(
        accept_features,
        accept_labels,
        train_size=DATASET_TRAIN_RATIO,
    )

    linear_model = train_linear_regression(train_features, train_labels)
    linear_error = mean_absolute_error(test_labels, linear_model.predict(test_features))
    print(f'Linear regression model MAE: {linear_error:.4f}')


if __name__ == '__main__':
    from core.loader import load_train_dataset
    sandbox_regression(dataset=load_train_dataset())
