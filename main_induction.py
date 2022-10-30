from os.path import join
import pandas as pd

from core.preprocessing import is_categorical_column, separate_features_label, split_training_test
from core.model_induction import train_decision_tree, load_decision_tree
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, OUTPUT_DIR


def _extract_categorical_dataset(dataset):
    categorical_feature_names = [column_name for column_name in dataset.columns
        if is_categorical_column(dataset[column_name])]
    return dataset.loc[:,categorical_feature_names]


def perform_decision_tree_induction(dataset):
    features, label = separate_features_label(dataset, DATASET_LABEL_NAME)
    categorical_features = _extract_categorical_dataset(features)
    categorical_label = pd.Series(label).map(bool)

    (train_features, train_labels), (test_features, test_labels) = split_training_test(
        categorical_features,
        categorical_label,
        train_factor=DATASET_TRAIN_RATIO,
        shuffle=True,
    )

    export_path = join(OUTPUT_DIR, 'decision_tree.json')
    try:
        model = load_decision_tree(export_path)
        print(f'Loaded decision tree dump from {export_path}')
    except FileNotFoundError:
        model = train_decision_tree(train_features, train_labels)
        model.dump(export_path)
        print(f'Wrote decision tree dump to {export_path}')

    pred_labels = model.predict(test_features)
    test_labels = list(test_labels)

    num_rows = len(test_features)
    num_predicted_accepts = sum(pred_labels)
    num_observed_accepts = sum(test_labels)
    accuracy = sum([test_labels[i] == y_hat
        for i, y_hat in enumerate(pred_labels)]) / num_rows
    print(f'Prediction: {num_predicted_accepts}/{num_rows} claims accepted'
        f'\nActual:     {num_observed_accepts}/{num_rows} claims accepted'
        f'\nDecision tree induction model achieved {accuracy * 100:.2f}% accuracy.')


if __name__ == '__main__':
    from core.loader import load_train_dataset
    perform_decision_tree_induction(dataset=load_train_dataset())
