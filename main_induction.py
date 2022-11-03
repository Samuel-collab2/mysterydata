import pandas as pd
from dataclasses import dataclass
from itertools import product

from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from core.preprocessing import separate_features_label, split_training_test, expand_dataset, \
    convert_label_boolean
from core.model_induction import NullDecisionTreeInduction
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, \
    SIGNIFICANT_BINARY_LABEL_FEATURES, SIGNIFICANT_FORWARD_STEPWISE_FEATURES


@dataclass
class ModelPerformance:
    train_accuracy: float
    test_accuracy: float
    test_precision: float
    test_recall: float


models = [
    (NullDecisionTreeInduction, {}),
    (DecisionTreeClassifier, {}),
    (DecisionTreeClassifier, {'max_depth': 20}),
    (RandomForestClassifier, {'n_estimators': 30}),
    (RandomForestClassifier, {'n_estimators': 30, 'max_depth': 20}),
    (KNeighborsClassifier, {'n_neighbors': 3}),
]

modifiers = {
    'feature_subset': (SIGNIFICANT_BINARY_LABEL_FEATURES,
        SIGNIFICANT_BINARY_LABEL_FEATURES[:3],
        SIGNIFICANT_FORWARD_STEPWISE_FEATURES,
        SIGNIFICANT_FORWARD_STEPWISE_FEATURES[:3]),
    'balance': (False, True),
}


def _format_kwargs(**kwargs):
    return ', '.join([f'{key}={value}'
        for key, value in kwargs.items()])

def _balance_binary_dataset(train_features, train_labels):
    dataset_label_name = train_labels.name

    train_samples = pd.concat((train_features, train_labels), axis='columns')

    true_samples = train_samples[train_samples[dataset_label_name] == True]
    false_samples = train_samples[train_samples[dataset_label_name] == False]
    min_samples = min(len(true_samples), len(false_samples))

    true_samples = true_samples[:min_samples]
    false_samples = false_samples[:min_samples]
    train_samples = pd.concat((true_samples, false_samples))

    print(f'Balance training set with {min_samples} accepted samples and {min_samples} rejected samples')
    train_features, train_labels = separate_features_label(train_samples, dataset_label_name)
    return train_features, train_labels


def perform_decision_tree_induction(dataset):
    features, labels = separate_features_label(dataset, DATASET_LABEL_NAME)
    features_expanded = expand_dataset(features)
    labels_boolean = convert_label_boolean(labels)

    (train_features, train_labels), (test_features, test_labels) = split_training_test(
        features_expanded,
        labels_boolean,
        train_factor=DATASET_TRAIN_RATIO,
        shuffle=True,
        seed=0,
    )

    categorical_columns = [column
        for column in features_expanded.columns
            if features_expanded[column].dtype == "uint8"]


    benchmark = None

    def evaluate_classifier(model, feature_subset=features_expanded.columns, balance=False):
        if balance:
            X_train, y_train = _balance_binary_dataset(
                train_features.loc[:, feature_subset],
                train_labels
            )
        else:
            X_train, y_train = (
                train_features.loc[:, feature_subset],
                train_labels,
            )

        model.fit(X_train, y_train)
        return evaluate_model(
            model,
            X_train,
            y_train,
            test_features.loc[:, feature_subset],
            test_labels,
            benchmark,
        )


    print('\nEvaluating performance of null induction model...')
    benchmark = evaluate_classifier(NullDecisionTreeInduction())

    for model_type, model_args in models:
        for modifier_values in product(*modifiers.values()):
            model = model_type(**model_args)
            eval_args = dict(zip(modifiers.keys(), modifier_values))
            formatted_args = _format_kwargs(**model_args, **eval_args)
            print(f'\nEvaluate {model_type.__name__}({formatted_args})')
            evaluate_classifier(model, **eval_args)


def _get_delta_tag(benchmark, value):
    delta = value - (benchmark or 0)
    delta_tag = (f' ({"+" if delta >= 0 else ""}{delta * 100:.2f}%)'
        if benchmark is not None
        else '')
    return delta_tag


def evaluate_model(model, train_features, train_labels, test_features, test_labels, benchmark=None):
    pred_labels = model.predict(test_features)
    test_labels = list(test_labels)

    num_rows = len(test_features)
    num_predicted_accepts = sum(pred_labels)
    num_observed_accepts = sum(test_labels)

    train_accuracy = model.score(train_features, train_labels)
    test_accuracy = model.score(test_features, test_labels)
    test_precision = precision_score(test_labels, pred_labels, zero_division=0)
    test_recall = recall_score(test_labels, pred_labels)

    print(f'Prediction:     {num_predicted_accepts}/{num_rows} claims accepted'
        f'\nActual:         {num_observed_accepts}/{num_rows} claims accepted'
        f'\nTrain accuracy: {train_accuracy * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.train_accuracy, train_accuracy) +
        f'\nTest accuracy:  {test_accuracy * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.test_accuracy, test_accuracy) +
        f'\nTest precision: {test_precision * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.test_precision, test_precision) +
        f'\nTest recall:    {test_recall * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.test_recall, test_recall))

    return ModelPerformance(
        train_accuracy,
        test_accuracy,
        test_precision,
        test_recall
    )


if __name__ == '__main__':
    from core.loader import load_train_dataset
    perform_decision_tree_induction(dataset=load_train_dataset())
