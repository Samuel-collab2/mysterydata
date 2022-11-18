from os.path import join
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek

from core.loader import load_determining_dataset
from core.preprocessing import separate_features_label, split_training_test, \
    convert_label_boolean, get_categorical_columns, expand_dataset_deterministic, \
    balance_binary_dataset, create_augmented_features
from core.model_induction import NullBinaryClassifier
from core.model_induction_nn import NeuralNetworkClassifier
from core.model_induction_wrapper import ModelInductionWrapper
from core.constants import OUTPUT_DIR, DATASET_LABEL_NAME, DATASET_TRAIN_RATIO
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS


NUM_BEST_MODELS = 10
NUM_K_FOLD_SPLITS = 3


@dataclass
class ModelPerformance:
    train_accuracy: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float


models = [
    (NeuralNetworkClassifier, {
        'epochs': 10,
        'batch_size': 100,
        'class_weight': {
            False: 0.5,
            True: 5,
        },
        'verbose': 1,
    }),
    (RandomForestClassifier, {
        'n_estimators': 50,
        'class_weight': 'balanced_subsample',
        'bootstrap': False
    }),
]

# model modifiers: all combinations are considered
modifiers = {
    'feature_subset': [
        ('augmented', SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS),
    ],
    'smote': (False, True),
    'pca': (0, 10),
}


def _format_kwargs(**kwargs):
    return ', '.join([f'{key}={value}'
        for key, value in kwargs.items()])

def _should_accept_claim(claim):
    return (claim['feature11'] == 5
        or claim['feature9'] == 0
        or claim['feature13'] == 4
        or claim['feature14'] == 3
        or claim['feature18'] == 1)

def _should_reject_claim(claim):
    return claim['feature7'] == 3


def _expand_features(features, determining_data=load_determining_dataset()):
    determining_features, _ = separate_features_label(determining_data, DATASET_LABEL_NAME)
    categorical_columns = get_categorical_columns(features)
    features_expanded = expand_dataset_deterministic(features, determining_features, categorical_columns)
    features_augmented = create_augmented_features(features, SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS)
    return pd.concat((features_expanded, features_augmented), axis='columns')


def perform_induction_tests(train_data, test_data):
    print('Running induction test suite...')

    train_features, train_labels = separate_features_label(train_data, DATASET_LABEL_NAME)
    train_features, train_labels = _expand_features(train_features), convert_label_boolean(train_labels)

    def evaluate_classifier(model, x_train, y_train, x_test, y_test,
                            feature_subset=train_features.columns,
                            rejection_skew=0,
                            wrap_induction=False,
                            smote=False,
                            pca=0,
                            benchmark=None):
        x_train = x_train[feature_subset]
        x_test = x_test[feature_subset]

        if rejection_skew:
            x_train, y_train = balance_binary_dataset(
                x_train,
                y_train,
                skew_false=rejection_skew,
            )

        if wrap_induction:
            model = ModelInductionWrapper(model,
                predicate_accept=_should_accept_claim,
                predicate_reject=_should_reject_claim)

        if (smote
        and not isinstance(model, NullBinaryClassifier)):
            x_train, y_train = SMOTETomek().fit_resample(x_train, y_train)

        if pca:
            analyzer = PCA(n_components=pca)
            x_train = pd.DataFrame(analyzer.fit_transform(x_train),
                columns=range(pca))
            x_test = pd.DataFrame(analyzer.fit_transform(x_test),
                columns=range(pca))

        if isinstance(model, NeuralNetworkClassifier):
            scaler = StandardScaler()
            x_train = pd.DataFrame(scaler.fit_transform(x_train),
                columns=x_train.columns)
            x_test = pd.DataFrame(scaler.fit_transform(x_test),
                columns=x_test.columns)
            model.fit(x_train, y_train,
                validation_data=(x_test, y_test))
        else:
            model.fit(x_train, y_train)

        return evaluate_model(
            model,
            x_train, y_train,
            x_test, y_test,
            benchmark,
        )


    print('\nEvaluating performance of null induction model...')
    (x_train, y_train), (x_test, y_test) = split_training_test(
        train_features,
        train_labels,
        train_factor=DATASET_TRAIN_RATIO,
        shuffle=True,
        seed=0,
    )
    null_benchmark = evaluate_classifier(NullBinaryClassifier(),
        x_train, y_train,
        x_test, y_test)


    # HACK: add full column sets at runtime
    # modifiers['feature_subset'].insert(0,
    #     ('all', list(train_features.columns)))


    def score_model(model):
        _, (model_f1, model_overfit) = model
        return model_f1 - model_overfit

    def sorted_models(model_scores):
        return sorted(model_scores.items(),
            key=score_model,
            reverse=True)


    model_scores = {}
    kfold = StratifiedKFold(n_splits=NUM_K_FOLD_SPLITS)

    modifier_combinations = product(*modifiers.values())
    for (model_type, model_args), modifier_values in product(models, modifier_combinations):
        model_results = []

        for k, (train_index, test_index) in enumerate(kfold.split(train_features, train_labels)):
            x_train, x_test = train_features.iloc[train_index], train_features.iloc[test_index]
            y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

            model = model_type(**model_args)
            model_modifiers = dict(zip(modifiers.keys(), modifier_values))

            formatted_args = _format_kwargs(**model_args, **{
                # HACK: use feature subset label for print-formatting
                key: (value[0] if key == 'feature_subset' else value)
                    for key, value in model_modifiers.items()
            })
            model_id = f'{model_type.__name__}({formatted_args})'
            print(f'\n{k+1}. Evaluate {model_id}')

            model_performance = evaluate_classifier(model,
                x_train, y_train,
                x_test, y_test,
                benchmark=null_benchmark,
                # HACK: use feature subset sequence for classification
                **{key: (value[1] if key == 'feature_subset' else value)
                    for key, value in model_modifiers.items()}
            )
            model_results.append(model_performance)

        model_scores[model_id] = (
            np.mean([result.test_f1 for result in model_results]),
            np.mean([abs(result.train_accuracy - result.test_accuracy) for result in model_results]),
        )  # absolute accuracy delta penalizes both overfit and underfit

        best_model_scores = sorted_models(model_scores)
        _write_model_rankings(_format_model_rankings(best_model_scores))

    print(f'\n-- Top {NUM_BEST_MODELS} model F1 scores (offset by overfit) --')
    print(_format_model_rankings(best_model_scores[:NUM_BEST_MODELS]))


def _format_model_score(model_score):
    model_f1, model_overfit = model_score
    return (f'({model_f1 * 100:.2f}'
        f'{"+" if model_overfit < 0 else "-"}'
        f'{abs(model_overfit) * 100:.1f})%')

def _format_model_rankings(model_rankings):
    return '\n'.join([f'{i + 1}. {_format_model_score(model_score)} :: {model_id}'
        for i, (model_id, model_score) in enumerate(model_rankings)])

def _write_model_rankings(rankings_buffer):
    rankings_path = join(OUTPUT_DIR, 'model_rankings.md')
    with open(rankings_path, mode='w', encoding='utf-8') as rankings_file:
        rankings_file.write(rankings_buffer)


def _get_delta_tag(benchmark, value):
    if (benchmark is None
    or benchmark == 0
    or value == benchmark):
        return ''
    delta = value - (benchmark or 0)
    return f' ({"+" if delta >= 0 else ""}{delta * 100:.2f}%)'


def evaluate_model(model, train_features, train_labels, test_features, test_labels, benchmark=None):
    pred_labels = list(model.predict(test_features))
    test_labels = list(test_labels)

    num_rows = len(test_features)
    num_predicted_accepts = int(sum(pred_labels))
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
    from core.loader import load_train_dataset, load_test_dataset
    print('Loading data for induction test suite...')
    perform_induction_tests(
        train_data=load_train_dataset(),
        test_data=load_test_dataset(),
    )
