import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from core.preprocessing import separate_features_label, get_categorical_columns, \
    expand_dataset_deterministic, convert_label_boolean, create_augmented_features, \
    balance_binary_dataset
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_COLUMNS


NUM_EPOCHS = 10


def get_induction_data(train_data, test_data):
    train_features, train_labels = separate_features_label(train_data, DATASET_LABEL_NAME)
    test_features = test_data

    categorical_columns = get_categorical_columns(train_features)
    categorical_train_features = train_features.loc[:, categorical_columns]
    categorical_test_features = test_features.loc[:, categorical_columns]

    combined_features = pd.concat((train_features, test_features))
    combined_expanded_features = expand_dataset_deterministic(
        combined_features,
        train_features,
        categorical_columns,
    )

    train_size = len(train_features)
    expanded_train_features = combined_expanded_features.iloc[:train_size]
    expanded_test_features = combined_expanded_features.iloc[train_size:]

    augmented_train_features = create_augmented_features(train_features, SIGNIFICANT_AUGMENTED_COLUMNS)
    augmented_test_features = create_augmented_features(test_features, SIGNIFICANT_AUGMENTED_COLUMNS)

    processed_train_features = pd.concat((
        expanded_train_features,
        augmented_train_features,
        categorical_train_features,
    ), axis='columns')

    processed_test_features = pd.concat((
        expanded_test_features,
        augmented_test_features,
        categorical_test_features,
    ), axis='columns')

    processed_train_labels = convert_label_boolean(train_labels)
    return (
        processed_train_features,
        processed_train_labels,
        processed_test_features,
    )


def run_simulation(train_features, train_labels, model=None, epoch=0):
    x_train, x_test, y_train, y_test = train_test_split(
        train_features,
        train_labels,
        train_size=DATASET_TRAIN_RATIO,
    )
    x_train, y_train = balance_binary_dataset(x_train, y_train, skew_false=6)

    if model is None:
        model = RandomForestClassifier(n_estimators=50)
        model.fit(x_train, y_train)

    evaluate_model(model, x_train, x_test, y_train, y_test, epoch=epoch)
    return model

def evaluate_model(model, x_train, x_test, y_train, y_test, epoch):
    print(f'\n-- Epoch {epoch}/{NUM_EPOCHS}')
    print(f'Training accuracy:\t{model.score(x_train, y_train)*100:.4f}%')
    print(f'Validation accuracy:\t{model.score(x_test, y_test)*100:.4f}%')
    print(f'Training F1:\t\t{f1_score(y_train, model.predict(x_train))*100:.4f}%')
    print(f'Validation F1:\t\t{f1_score(y_test, model.predict(x_test))*100:.4f}%')

def sandbox_iterative_induction(train_data, test_data):
    train_features, train_labels, test_features = get_induction_data(train_data, test_data)
    train_features = train_features[SIGNIFICANT_AUGMENTED_COLUMNS]
    test_features = test_features[SIGNIFICANT_AUGMENTED_COLUMNS]

    model = run_simulation(train_features, train_labels, epoch=1)

    for epoch in range(2, NUM_EPOCHS):
        pred_labels = model.predict(test_features)
        pred_labels = pd.Series(pred_labels, name=DATASET_LABEL_NAME)
        model = run_simulation(
            pd.concat((train_features, test_features)),
            pd.concat((train_labels, pred_labels)),
            epoch=epoch
        )

    run_simulation(train_features, train_labels, model, epoch=NUM_EPOCHS)


if __name__ == '__main__':
    from core.loader import load_train_dataset, load_test_dataset
    print('Loading data for induction test suite...')
    sandbox_iterative_induction(
        train_data=load_train_dataset(),
        test_data=load_test_dataset(),
    )
