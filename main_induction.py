from os.path import join

from core.preprocessing import split_training_test, preprocess_induction_data, separate_features_label
from core.model_induction import NullDecisionTreeInduction, BinaryDecisionTreeInduction, train_decision_tree
from core.constants import DATASET_TRAIN_RATIO, OUTPUT_DIR, DATASET_LABEL_NAME
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from core.preprocessing import separate_features_label, split_training_test, \
    expand_dataset
from core.model_induction import NullDecisionTreeInduction
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, \
    SIGNIFICANT_BINARY_LABEL_FEATURES, SIGNIFICANT_FORWARD_STEPWISE_FEATURES


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

    categorical_columns = [column
        for column in features_expanded.columns
            if features_expanded[column].dtype == "uint8"]


    def evaluate_classifier(model, feature_subset=features_expanded.columns, accuracy_benchmark=None):
        model.fit(train_features.loc[:, feature_subset], train_labels)
        return evaluate_model(model, test_features.loc[:, feature_subset], test_labels, accuracy_benchmark)


    print('\nEvaluating performance of null induction model...')
    accuracy = evaluate_classifier(NullDecisionTreeInduction())

    print('\nEvaluating performance of base DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), accuracy_benchmark=accuracy)

    print('\nEvaluating performance of entropy-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), accuracy_benchmark=accuracy)

    print('\nEvaluating performance of categorical DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), categorical_columns, accuracy_benchmark=accuracy)

    print('\nEvaluating performance of categorical entropy-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), categorical_columns, accuracy_benchmark=accuracy)

    print('\nEvaluating performance of most significant binary label feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_BINARY_LABEL_FEATURES, accuracy_benchmark=accuracy)

    print('\nEvaluating performance of 7 most significant binary label feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_BINARY_LABEL_FEATURES[:7], accuracy_benchmark=accuracy)

    print('\nEvaluating performance of 3 most significant binary label feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_BINARY_LABEL_FEATURES[:3], accuracy_benchmark=accuracy)

    print('\nEvaluating performance of most significant forward stepwise feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_FORWARD_STEPWISE_FEATURES, accuracy_benchmark=accuracy)

    print('\nEvaluating performance of 7 most significant forward stepwise feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_FORWARD_STEPWISE_FEATURES[:7], accuracy_benchmark=accuracy)

    print('\nEvaluating performance of 3 most significant forward stepwise feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_FORWARD_STEPWISE_FEATURES[:3], accuracy_benchmark=accuracy)


def evaluate_model(model, test_features, test_labels, accuracy_benchmark=None):
    pred_labels = model.predict(test_features)
    test_labels = list(test_labels)

    num_rows = len(test_features)
    num_predicted_accepts = sum(pred_labels)
    num_observed_accepts = sum(test_labels)
    accuracy = accuracy_score(test_labels, pred_labels)
    accuracy_delta = accuracy - (accuracy_benchmark or 0)
    accuracy_delta_tag = (f' ({"+" if accuracy_delta >= 0 else ""}{accuracy_delta * 100:.2f}%)'
        if accuracy_benchmark is not None
        else '')

    print(f'Prediction: {num_predicted_accepts}/{num_rows} claims accepted'
        f'\nActual:     {num_observed_accepts}/{num_rows} claims accepted'
        f'\nAccuracy:   {accuracy * 100:.2f}%{accuracy_delta_tag}')

    return accuracy


if __name__ == '__main__':
    from core.loader import load_train_dataset
    perform_decision_tree_induction(dataset=load_train_dataset())
