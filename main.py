from core.loader import load_train_dataset
from core.model import handle_linear_regression, calculate_mae
from core.processing import expand_dataset, split_data_training_test, split_dataset_features_label
from core.plot import handle_basic_plots, handle_compound_plots
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO


def main():
    dataset_raw = load_train_dataset()
    data_raw = split_dataset_features_label(dataset_raw, DATASET_LABEL_NAME)

    handle_basic_plots(data_raw, DATASET_LABEL_NAME)
    handle_compound_plots(data_raw, DATASET_LABEL_NAME)

    dataset_expanded = expand_dataset(dataset_raw)
    data_expanded = split_dataset_features_label(dataset_expanded, DATASET_LABEL_NAME)

    train_features, test_features, train_label, test_label = split_data_training_test(
        data_expanded,
        train_factor=DATASET_TRAIN_RATIO
    )

    predictions = handle_linear_regression(train_features, train_label, test_features)
    mae = calculate_mae(test_label, predictions)
    print(f'Linear regression MAE: {mae:.4f}')


if __name__ == '__main__':
    main()
