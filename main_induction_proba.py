import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, \
    accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.combine import SMOTEENN

from core.preprocessing import get_induction_data, separate_features_label
from core.constants import DATASET_TRAIN_RATIO, DATASET_LABEL_NAME
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS
from core.model_base import BaseModel


CLASS_WEIGHT = {
    False: 1,
    True: 20,
}

models = [
    (RandomForestClassifier, {
        'n_estimators': 50,
        'class_weight': {False: 1, True: 15},
    }),
    (RandomForestClassifier, {
        'n_estimators': 50,
        'class_weight': {False: 1, True: 20},
        'oob_score': True,
    }),
    (RandomForestClassifier, {
        'n_estimators': 50,
        'class_weight': {False: 1, True: 25},
        'bootstrap': False,
    }),
]

NUM_SPLITS = len(models)
DECISION_BOUNDARY = 0.5


class SkepticalBinaryClassifier(BaseModel):
    DECISION_BOUNDARY = 0.95

    def __init__(self, estimators):
        self._estimators = estimators

    def fit(self, train_features, train_labels):
        pass

    def predict(self, test_features):
        _, true_proba = zip(*self.predict_proba(test_features))
        return np.array([p > self.DECISION_BOUNDARY for p in true_proba])

    def predict_proba(self, test_features):
        probas = [model.predict_proba(test_features)
            for model in self._estimators]

        aggregate_proba = probas[0]
        for proba in probas[1:]:
            aggregate_proba *= proba

        return aggregate_proba

    def score(self, test_features, test_labels):
        return self._estimators[0].score(test_features, test_labels)


class IsolationForestClassifier(BaseModel):

    def __init__(self, *args, **kwargs):
        self._model = IsolationForest(*args, **kwargs)

    def fit(self, train_features, train_labels):
        self._model.fit(train_features, train_labels)

    def predict(self, test_features):
        return self._model.predict(test_features) == -1

    def score(self, test_features, test_labels):
        return accuracy_score(test_labels, self.predict(test_features))


def create_model(train_features=None, train_labels=None, **kwargs):
    model = RandomForestClassifier(**{
        'n_estimators': 50,
        'class_weight': CLASS_WEIGHT,
        'bootstrap': False,
        **kwargs,
    })
    if train_features is not None and train_labels is not None:
        model.fit(train_features, train_labels)
    return model

def evaluate_model(model, train_features, train_labels, test_features, test_labels):
    pred_labels = model.predict(test_features)
    print(f'Train F1:        {f1_score(train_labels, model.predict(train_features))*100:.2f}%')
    print(f'Test F1:         {f1_score(test_labels, pred_labels)*100:.2f}%')
    print(f'Test precision:  {precision_score(test_labels, pred_labels)*100:.2f}%')
    print(f'Test recall:     {recall_score(test_labels, pred_labels)*100:.2f}%')

    tn, fp, fn, tp = confusion_matrix(test_labels, model.predict(test_features)).ravel()
    print(f'True positives:  {tp}/{(tp+fp)} ({tp/(tp+fp)*100:.2f}%)')
    print(f'True negatives:  {tn}/{(tn+fn)} ({tn/(tn+fn)*100:.2f}%)')
    print(f'False positives: {fp}/{(tp+fp)} ({fp/(tp+fp)*100:.2f}%)')
    print(f'False negatives: {fn}/{(tn+fn)} ({fn/(tn+fn)*100:.2f}%)')

def evaluate_model_with_confusion(model, train_features, train_labels, test_features, test_labels, decision_boundary=0.5):
    pred_proba = model.predict_proba(test_features)
    _, accept_proba = zip(*pred_proba)
    pred_labels = model.predict(test_features)

    print(f'Train F1:\t{f1_score(train_labels, model.predict(train_features))*100:.2f}%')
    print(f'Test F1:\t{f1_score(test_labels, pred_labels)*100:.2f}%')
    print(f'Test precision:\t{precision_score(test_labels, pred_labels)*100:.2f}%')
    print(f'Test recall:\t{recall_score(test_labels, pred_labels)*100:.2f}%')

    num_true_positives = sum(test_labels)
    num_true_negatives = len(test_labels) - num_true_positives
    num_true_positives_found = 0
    num_true_negatives_found = 0

    false_positive_proba = []
    num_false_negatives_found = 0
    num_sure_false_positives_found = 0
    num_sure_false_negatives_found = 0

    for i, p in enumerate(accept_proba):
        v = bool(test_labels.iloc[i])

        if p <= decision_boundary and v is False:
            num_true_negatives_found += 1
            continue

        if p > decision_boundary and v is True:
            num_true_positives_found += 1
            continue

        if p == 1 and v is True:
            num_sure_false_negatives_found += 1
            continue

        if p == 1 and v is False:
            num_sure_false_positives_found += 1
            continue

        if p <= decision_boundary and v is True:
            num_false_negatives_found += 1
            continue

        if p > decision_boundary and v is False:
            false_positive_proba.append(p)
            continue

    num_false_positives_found = len(false_positive_proba)
    num_all_positives_found = num_true_positives_found + num_false_positives_found
    num_all_negatives_found = num_true_negatives_found + num_false_negatives_found

    print(f'Found {num_true_positives_found}/{num_true_positives} true positive(s)'
        f' ({num_true_positives_found/num_true_positives*100:.2f}%)')
    print(f'Found {num_true_negatives_found}/{num_true_negatives} true negative(s)'
        f' ({num_true_negatives_found/num_true_negatives*100:.2f}%)')
    print(f'Encountered {num_false_positives_found} false positive(s)'
        f' ({num_false_positives_found/(num_all_positives_found)*100:.2f}% of positives incorrectly classified'
        f' with mean probability {np.mean(false_positive_proba)*100:.2f}%)')
    print(f'Encountered {num_false_negatives_found} false negative(s)'
        f' ({num_false_negatives_found/(num_all_negatives_found)*100:.2f}% of negatives incorrectly classified)')
    print(f'{num_sure_false_positives_found} of positive(s) were actually false despite 100% probability'
        f' ({num_sure_false_positives_found/(num_all_positives_found)*100:.2f}% of positives incorrectly classified)')
    print(f'{num_sure_false_negatives_found} of negative(s) were actually true despite 100% probability'
        f' ({num_sure_false_negatives_found/(num_all_negatives_found)*100:.2f}% of negatives incorrectly classified)')


def sandbox_induction_skeptical_classifier(train_data, test_data):
    print('Running skeptical classifier sandbox...')

    features, labels, _ = get_induction_data(train_data, test_data)
    features = features[SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        train_size=DATASET_TRAIN_RATIO,
    )

    train_data = pd.concat((train_features, train_labels), axis='columns')
    accept_data = train_data[train_data[DATASET_LABEL_NAME] == True]
    reject_data = train_data[train_data[DATASET_LABEL_NAME] == False]

    accept_features, accept_labels = separate_features_label(accept_data, DATASET_LABEL_NAME)
    reject_features, reject_labels = separate_features_label(reject_data, DATASET_LABEL_NAME)

    estimators = []
    kfold = KFold(n_splits=NUM_SPLITS)
    for k, (_, reject_index) in enumerate(kfold.split(reject_features)):
        print(f'\n-- Model {k+1}/{NUM_SPLITS}')
        x_train = pd.concat((reject_features.iloc[reject_index], accept_features))
        y_train = pd.concat((reject_labels.iloc[reject_index], accept_labels))
        model = create_model(x_train, y_train)
        estimators.append(model)
        evaluate_model(model, x_train, y_train, test_features, test_labels)

    print('\n-- Benchmark model')
    model = create_model(train_features, train_labels)
    evaluate_model_with_confusion(model, train_features, train_labels, test_features, test_labels)

    print('\n-- Final model')
    model = SkepticalBinaryClassifier(estimators)
    model.fit(train_features, train_labels)
    evaluate_model_with_confusion(model, train_features, train_labels, test_features, test_labels,
       decision_boundary=model.DECISION_BOUNDARY)

def sandbox_induction_resampler(train_data, test_data):
    print('Running resampler sandbox...')

    features, labels, _ = get_induction_data(train_data, test_data)
    features = features[SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        train_size=DATASET_TRAIN_RATIO,
    )

    print('\n-- Training random forest...')
    random_forest = create_model(train_features, train_labels, random_state=0)
    evaluate_model(random_forest, train_features, train_labels, test_features, test_labels)

    print('\n-- Resampling dataset...')
    resampler = SMOTEENN()
    resampled_features, resampled_labels = resampler.fit_resample(train_features, train_labels)

    print(f'Dataset size delta: {len(resampled_features) - len(train_features)}')
    train_features, train_labels = resampled_features, resampled_labels

    print('\n-- Training resampled random forest...')
    random_forest = create_model(train_features, train_labels, random_state=0)
    evaluate_model(random_forest, train_features, train_labels, test_features, test_labels)

def sandbox_induction_isolation_forest(train_data, test_data):
    print('Running isolation forest sandbox...')

    features, labels, _ = get_induction_data(train_data, test_data)
    features = features[SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        train_size=DATASET_TRAIN_RATIO,
    )

    print('\n-- Training isolation forest classifier...')
    isolation_forest = IsolationForestClassifier()
    isolation_forest.fit(train_features, train_labels)
    evaluate_model(isolation_forest, train_features, train_labels, test_features, test_labels)

    train_outliers = isolation_forest.predict(train_features)
    unproblematic_indices = [i for i, v in enumerate(train_outliers) if not v]
    train_features = train_features.iloc[unproblematic_indices]
    train_labels = train_labels.iloc[unproblematic_indices]
    print(f'Removed {len(train_outliers) - len(unproblematic_indices)} outliers')

    print('\n-- Training random forest classifier...')
    random_forest = create_model(train_features, train_labels)
    evaluate_model(random_forest, train_features, train_labels, test_features, test_labels)

    pred_proba = list(zip(*random_forest.predict_proba(test_features)))[1]
    pred_outlier_indices = {i for i, v in enumerate(isolation_forest.predict(test_features)) if v}
    test_label_indices = {i for i, v in enumerate(test_labels) if v}
    positive_indices = {i for i, p in enumerate(pred_proba) if p > 0.5}
    sure_positive_indices = {i for i, p in enumerate(pred_proba) if p == 1}
    false_positive_indices = positive_indices - test_label_indices
    sure_false_positive_indices = sure_positive_indices - test_label_indices
    sus_false_positives = sure_false_positive_indices & pred_outlier_indices
    sus_true_positives = test_label_indices & pred_outlier_indices
    print(f'\n{len(false_positive_indices)} false positives')
    print(f'{len(sure_false_positive_indices)} sure positives were actually false')
    print(f'Isolation forest was suspicious of {len(sus_false_positives)} false positives')
    print(f'Isolation forest was suspicious of {len(sus_true_positives)} true positives')

def sandbox_induction_threshold_moving(train_data, test_data):
    print('Running threshold moving sandbox...')

    features, labels, _ = get_induction_data(train_data, test_data)
    features = features[SIGNIFICANT_AUGMENTED_INDUCTION_COLUMNS]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        train_size=DATASET_TRAIN_RATIO,
    )

    print('\n-- Training random forest classifier...')
    random_forest = create_model(train_features, train_labels)
    evaluate_model(random_forest, train_features, train_labels, test_features, test_labels)

    def optimize_threshold(test_labels, pred_proba, metric=f1_score):
        thresholds = np.arange(0, 1, 0.001)
        scores = [metric(test_labels, pred_proba >= threshold)
            for threshold in thresholds]
        return thresholds[np.argmax(scores)]

    print('\n-- Searching for optimal training probability threshold for max precision on training data...')
    train_proba = random_forest.predict_proba(train_features)[:, 1]
    train_threshold = optimize_threshold(train_labels, train_proba, metric=precision_score)
    train_score = f1_score(train_labels, train_proba >= train_threshold)
    print(f'Best training threshold is {train_threshold} with training F1 score {train_score*100:.2f}%')

    pred_proba = random_forest.predict_proba(test_features)[:, 1]
    pred_score = f1_score(test_labels, pred_proba >= train_threshold)
    print(f'Test F1 score given training threshold is {pred_score*100:.2f}%')

    test_threshold = optimize_threshold(test_labels, pred_proba)
    test_score = f1_score(test_labels, pred_proba >= test_threshold)
    print(f'Actual best threshold was {test_threshold:.3f} with test F1 score {test_score*100:.2f}%')

if __name__ == '__main__':
    from core.loader import load_train_dataset, load_test_dataset
    sandbox_induction_resampler(
        train_data=load_train_dataset(),
        test_data=load_test_dataset(),
    )