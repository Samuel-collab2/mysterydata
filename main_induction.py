import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from core.preprocessing import separate_features_label, split_training_test, \
    expand_dataset
from core.model_induction import NullDecisionTreeInduction
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO


def is_categorical_column(column):
    return column.dtype == "uint8"

def _extract_categorical_dataset(dataset):
    categorical_feature_names = [column_name for column_name in dataset.columns
        if is_categorical_column(dataset[column_name])]
    return dataset.loc[:, categorical_feature_names]


def perform_decision_tree_induction(dataset):
    features, labels = separate_features_label(dataset, DATASET_LABEL_NAME)
    features_expanded = expand_dataset(features)
    labels_binary = pd.Series(labels).map(bool)

    (train_features, train_labels), (test_features, test_labels) = split_training_test(
        features_expanded,
        labels_binary,
        train_factor=DATASET_TRAIN_RATIO,
        shuffle=True,
    )

    print('\nEvaluating performance of null induction model...')
    evaluate_model(NullDecisionTreeInduction(), test_features, test_labels)

    print('\nEvaluating performance of base DecisionTreeClassifier...')
    model = DecisionTreeClassifier()
    model.fit(train_features, train_labels)
    evaluate_model(model, test_features, test_labels)

    print('\nEvaluating performance of entropy-based DecisionTreeClassifier...')
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(train_features, train_labels)
    evaluate_model(model, test_features, test_labels)

    train_features = _extract_categorical_dataset(train_features)
    test_features = _extract_categorical_dataset(test_features)

    print('\nEvaluating performance of categorical DecisionTreeClassifier...')
    model = DecisionTreeClassifier()
    model.fit(train_features, train_labels)
    evaluate_model(model, test_features, test_labels)

    print('\nEvaluating performance of categorical entropy-based DecisionTreeClassifier...')
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(train_features, train_labels)
    evaluate_model(model, test_features, test_labels)


def evaluate_model(model, test_features, test_labels):
    pred_labels = model.predict(test_features)
    test_labels = list(test_labels)

    num_rows = len(test_features)
    num_predicted_accepts = sum(pred_labels)
    num_observed_accepts = sum(test_labels)
    accuracy = accuracy_score(test_labels, pred_labels)

    print(f'Prediction: {num_predicted_accepts}/{num_rows} claims accepted'
        f'\nActual:     {num_observed_accepts}/{num_rows} claims accepted'
        f'\nDecision tree induction model achieved {accuracy * 100:.2f}% accuracy.')


if __name__ == '__main__':
    from core.loader import load_train_dataset
    perform_decision_tree_induction(dataset=load_train_dataset())
