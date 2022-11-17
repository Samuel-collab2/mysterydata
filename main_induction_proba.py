from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from core.preprocessing import get_induction_data
from core.constants import DATASET_TRAIN_RATIO
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_COLUMNS


CONFIDENCE_THRESHOLD = 1 / 2


def sandbox_induction_proba(train_data, test_data):
    print('Running induction probability sandbox...')

    features, labels, _ = get_induction_data(train_data, test_data)
    features = features[SIGNIFICANT_AUGMENTED_COLUMNS]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        train_size=DATASET_TRAIN_RATIO,
    )

    model = RandomForestClassifier(
        n_estimators=50,
        class_weight='balanced_subsample',
        bootstrap=False,
    )
    model.fit(train_features, train_labels)
    print(f'Train F1:\t{f1_score(train_labels, model.predict(train_features))*100:.2f}%')
    print(f'Test F1:\t{f1_score(test_labels, model.predict(test_features))*100:.2f}%')

    pred_proba = model.predict_proba(test_features)
    _, accept_proba = zip(*pred_proba)

    num_true_positives = sum(test_labels)
    num_true_negatives = len(test_labels) - num_true_positives
    num_true_positives_found = 0
    num_true_negatives_found = 0
    num_false_positives_found = 0
    num_false_negatives_found = 0

    for i, p in enumerate(accept_proba):
        v = bool(test_labels.iloc[i])

        if p < CONFIDENCE_THRESHOLD and v is False:
            num_true_negatives_found += 1
            continue

        if p > 1 - CONFIDENCE_THRESHOLD and v is True:
            num_true_positives_found += 1
            continue

        if p < CONFIDENCE_THRESHOLD and v is True:
            num_false_negatives_found += 1
            continue

        if p > 1 - CONFIDENCE_THRESHOLD and v is False:
            num_false_positives_found += 1
            continue

    print(f'Found {num_true_positives_found}/{num_true_positives} true positive(s)'
        f' ({num_true_positives_found/num_true_positives*100:.2f}%)')
    print(f'Found {num_true_negatives_found}/{num_true_negatives} true negative(s)'
        f' ({num_true_negatives_found/num_true_negatives*100:.2f}%)')
    print(f'Encountered {num_false_positives_found} false positive(s)'
        f' ({num_false_positives_found/(num_true_positives_found+num_false_positives_found)*100:.2f}% of positives incorrectly classified)')
    print(f'Encountered {num_false_negatives_found} false negative(s)'
        f' ({num_false_negatives_found/(num_true_negatives_found+num_false_negatives_found)*100:.2f}% of negatives incorrectly classified)')


if __name__ == '__main__':
    from core.loader import load_train_dataset, load_test_dataset
    sandbox_induction_proba(
        train_data=load_train_dataset(),
        test_data=load_test_dataset(),
    )
