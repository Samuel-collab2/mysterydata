from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier

from core.preprocessing import separate_features_label, split_training_test, expand_dataset, \
    convert_label_boolean
from core.model_induction import NullDecisionTreeInduction
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, \
    SIGNIFICANT_BINARY_LABEL_FEATURES, SIGNIFICANT_FORWARD_STEPWISE_FEATURES


def perform_decision_tree_induction(dataset):
    features, labels = separate_features_label(dataset, DATASET_LABEL_NAME)
    features_expanded = expand_dataset(features)
    labels_boolean = convert_label_boolean(labels)

    (train_features, train_labels), (test_features, test_labels) = split_training_test(
        features_expanded,
        labels_boolean,
        train_factor=DATASET_TRAIN_RATIO,
        shuffle=True,
    )

    categorical_columns = [column
        for column in features_expanded.columns
            if features_expanded[column].dtype == "uint8"]


    accuracy, precision = None, None

    def evaluate_classifier(model, feature_subset=features_expanded.columns):
        model.fit(train_features.loc[:, feature_subset], train_labels)
        return evaluate_model(
            model,
            test_features.loc[:, feature_subset],
            test_labels,
            accuracy,
            precision,
        )


    print('\nEvaluating performance of null induction model...')
    accuracy, precision = evaluate_classifier(NullDecisionTreeInduction())

    print('\nEvaluating performance of base DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier())

    print('\nEvaluating performance of entropy-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(criterion='entropy'))

    print('\nEvaluating performance of categorical DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), categorical_columns)

    print('\nEvaluating performance of categorical entropy-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), categorical_columns)

    print('\nEvaluating performance of most significant binary label feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_BINARY_LABEL_FEATURES)

    print('\nEvaluating performance of 7 most significant binary label feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_BINARY_LABEL_FEATURES[:7])

    print('\nEvaluating performance of 3 most significant binary label feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_BINARY_LABEL_FEATURES[:3])

    print('\nEvaluating performance of most significant forward stepwise feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_FORWARD_STEPWISE_FEATURES)

    print('\nEvaluating performance of 7 most significant forward stepwise feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_FORWARD_STEPWISE_FEATURES[:7])

    print('\nEvaluating performance of 3 most significant forward stepwise feature-based DecisionTreeClassifier...')
    evaluate_classifier(DecisionTreeClassifier(), SIGNIFICANT_FORWARD_STEPWISE_FEATURES[:3])


def _get_delta_tag(benchmark, value):
    delta = value - (benchmark or 0)
    delta_tag = (f' ({"+" if delta >= 0 else ""}{delta * 100:.2f}%)'
        if benchmark is not None
        else '')
    return delta_tag


def evaluate_model(model, test_features, test_labels, accuracy_benchmark=None, precision_benchmark=None):
    pred_labels = model.predict(test_features)
    test_labels = list(test_labels)

    num_rows = len(test_features)
    num_predicted_accepts = sum(pred_labels)
    num_observed_accepts = sum(test_labels)

    accuracy = accuracy_score(test_labels, pred_labels)
    precision = precision_score(test_labels, pred_labels)

    print(f'Prediction: {num_predicted_accepts}/{num_rows} claims accepted'
        f'\nActual:     {num_observed_accepts}/{num_rows} claims accepted'
        f'\nAccuracy:   {accuracy * 100:.2f}%{_get_delta_tag(accuracy_benchmark, accuracy)}'
        f'\nPrecision:  {precision * 100:.2f}%{_get_delta_tag(precision_benchmark, precision)}')

    return accuracy, precision


if __name__ == '__main__':
    from core.loader import load_train_dataset
    perform_decision_tree_induction(dataset=load_train_dataset())
