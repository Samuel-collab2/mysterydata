from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from core.preprocessing import get_induction_data
from core.constants import DATASET_TRAIN_RATIO
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_COLUMNS


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
    pred_proba = model.predict_proba(test_features)
    _, true_proba = zip(*pred_proba)
    print(f'Train F1:\t{f1_score(train_labels, model.predict(train_features))*100:.4f}%')
    print(f'Test F1:\t{f1_score(test_labels, model.predict(test_features))*100:.4f}%')


if __name__ == '__main__':
    from core.loader import load_train_dataset, load_test_dataset
    sandbox_induction_proba(
        train_data=load_train_dataset(),
        test_data=load_test_dataset(),
    )
