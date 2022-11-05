from dataclasses import dataclass
from itertools import product
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from core.preprocessing import separate_features_label, split_training_test, expand_dataset, \
    convert_label_boolean
from core.model_induction import NullDecisionTreeInduction
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, \
    SIGNIFICANT_BINARY_LABEL_FEATURES, SIGNIFICANT_FORWARD_STEPWISE_FEATURES


NUM_BEST_MODELS = 10
REJECTION_SKEW = 4  # HACK: 4x rejected than accepted


@dataclass
class ModelPerformance:
    train_accuracy: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float


models = [
    (DecisionTreeClassifier, {}),
    (DecisionTreeClassifier, {'max_depth': 20}),
    (RandomForestClassifier, {'n_estimators': 30}),
    (RandomForestClassifier, {'n_estimators': 30, 'max_depth': 20}),
]

# model modifiers: all combinations are considered
modifiers = {
    'feature_subset': [
        ('all_ridge', SIGNIFICANT_BINARY_LABEL_FEATURES),
        ('top3_ridge', SIGNIFICANT_BINARY_LABEL_FEATURES[:3]),
        ('all_prop', SIGNIFICANT_FORWARD_STEPWISE_FEATURES),
        ('top3_prop', SIGNIFICANT_FORWARD_STEPWISE_FEATURES[:3]),
    ],
    'balance': (True, False),
}


def _format_kwarg_value(value):
    return (f'list[{len(value)}]'
        if isinstance(value, list)
        else value)

def _format_kwargs(**kwargs):
    return ', '.join([f'{key}={_format_kwarg_value(value)}'
        for key, value in kwargs.items()])

def _balance_binary_dataset(train_features, train_labels, skew_true=1, skew_false=1):
    """
    Balances a binary dataset (i.e. boolean labels).
    Skew paramaters are used to fine-tune bias.
    :param train_features: Training features
    :param train_labels: Training labels
    :param skew_true: Factor of true sample count in resulting dataset
    :param skew_false: Factor of false sample count in resulting dataset
    """

    dataset_label_name = train_labels.name

    train_samples = pd.concat((train_features, train_labels), axis='columns')

    true_samples = train_samples[train_samples[dataset_label_name] == True]
    false_samples = train_samples[train_samples[dataset_label_name] == False]
    min_samples = min(len(true_samples), len(false_samples))

    true_samples = true_samples[:min_samples * skew_true]
    false_samples = false_samples[:min_samples * skew_false]
    train_samples = pd.concat((true_samples, false_samples))

    print(f'Balance training set'
        f' with {min_samples} accepted samples'
        f' and {min_samples} rejected samples')
    train_features, train_labels = separate_features_label(train_samples, dataset_label_name)
    return train_features, train_labels


def perform_induction(dataset):
    print('Running induction test suite...')

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

    benchmark = None

    def evaluate_classifier(model, feature_subset=features_expanded.columns, balance=False):
        if balance:
            x_train, y_train = _balance_binary_dataset(
                train_features.loc[:, feature_subset],
                train_labels,
                skew_false=REJECTION_SKEW
            )
        else:
            x_train, y_train = (
                train_features.loc[:, feature_subset],
                train_labels,
            )

        model.fit(x_train, y_train)
        return evaluate_model(
            model,
            x_train,
            y_train,
            test_features.loc[:, feature_subset],
            test_labels,
            benchmark,
        )


    print('\nEvaluating performance of null induction model...')
    benchmark = evaluate_classifier(NullDecisionTreeInduction())


    # HACK: add full column set at runtime
    modifiers['feature_subset'].append(
        ('all_features', list(features_expanded.columns))
    )


    model_scores = {}
    for model_type, model_args in models:
        for modifier_values in product(*modifiers.values()):
            model = model_type(**model_args)
            eval_args = dict(zip(modifiers.keys(), modifier_values))
            formatted_args = _format_kwargs(**model_args, **{
                # HACK: use feature subset label for print-formatting
                key: (value[0] if key == 'feature_subset' else value)
                    for key, value in eval_args.items()
            })

            model_id = f'{model_type.__name__}({formatted_args})'
            print(f'\nEvaluate {model_id}')

            model_performance = evaluate_classifier(model, **{
                # HACK: use feature subset sequence for classification
                key: (value[1] if key == 'feature_subset' else value)
                    for key, value in eval_args.items()
            })
            model_scores[model_id] = model_performance.test_f1

    best_model_scores = sorted(model_scores.items(),
        key=lambda item: item[1],
        reverse=True)[:NUM_BEST_MODELS]

    print(f'\n-- Top {NUM_BEST_MODELS} model F1 scores --')
    print('\n'.join([f'{i + 1}. {model_score * 100:.2f}% -- {model_id}'
        for i, (model_id, model_score) in enumerate(best_model_scores)]))


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
    test_f1 = f1_score(test_labels, pred_labels)

    print(f'Prediction:     {num_predicted_accepts}/{num_rows} claims accepted'
        f'\nActual:         {num_observed_accepts}/{num_rows} claims accepted'
        f'\nTrain accuracy: {train_accuracy * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.train_accuracy, train_accuracy) +
        f'\nTest accuracy:  {test_accuracy * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.test_accuracy, test_accuracy) +
        f'\nTest precision: {test_precision * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.test_precision, test_precision) +
        f'\nTest recall:    {test_recall * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.test_recall, test_recall) +
        f'\nTest F1 score:  {test_f1 * 100:.2f}%'
            + _get_delta_tag(benchmark and benchmark.test_f1, test_f1))

    return ModelPerformance(
        train_accuracy,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
    )


if __name__ == '__main__':
    from core.loader import load_train_dataset
    print('Loading dataset...')
    perform_induction(dataset=load_train_dataset())
