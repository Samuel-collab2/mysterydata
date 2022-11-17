import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from core.preprocessing import separate_features_label, get_categorical_columns, \
    expand_dataset_deterministic, convert_label_boolean, create_augmented_features
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_COLUMNS


NUM_EPOCHS = 3


def _balance_dataset(features, labels):
    return features, labels

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


def create_model(x_train, y_train):
    model = RandomForestClassifier(
        n_estimators=50,
        class_weight='balanced_subsample',
        bootstrap=False,
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_train, y_train, x_test, y_test):
    f1 = f1_score(y_test, model.predict(x_test))
    print(f'Training accuracy:\t{model.score(x_train, y_train)*100:.4f}%')
    print(f'Validation accuracy:\t{model.score(x_test, y_test)*100:.4f}%')
    print(f'Training F1:\t\t{f1_score(y_train, model.predict(x_train))*100:.4f}%')
    print(f'Validation F1:\t\t{f1*100:.4f}%')
    return f1


def sandbox_iterative_induction(train_data, test_data):
    train_features, train_labels, valid_features = get_induction_data(train_data, test_data)
    train_features, valid_features = (
        train_features[SIGNIFICANT_AUGMENTED_COLUMNS],
        valid_features[SIGNIFICANT_AUGMENTED_COLUMNS],
    )

    train_features, test_features, train_labels, test_labels = train_test_split(
        train_features,
        train_labels,
        train_size=DATASET_TRAIN_RATIO,
    )

    model = create_model(*_balance_dataset(train_features, train_labels))

    print('\n-- Initial model')
    init_f1 = evaluate_model(model, train_features, train_labels, test_features, test_labels)

    best_f1 = init_f1
    best_model = model

    for epoch in range(0, NUM_EPOCHS):
        print(f'\n-- Epoch {epoch + 1}/{NUM_EPOCHS}')

        pred_labels = model.predict(valid_features)
        pred_labels = pd.Series(pred_labels, name=DATASET_LABEL_NAME)
        x_train, y_train = _balance_dataset(
            pd.concat((train_features, valid_features)),
            pd.concat((train_labels, pred_labels)),
        )
        x_test, y_test = test_features, test_labels

        model = create_model(x_train, y_train)
        f1 = evaluate_model(model, x_train, y_train, x_test, y_test)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print('\n-- Results')
    print(f'Best model has a validation F1 score of {best_f1*100:.4f}% (+{(best_f1-init_f1)*100:.2f}%)')
    return best_model


if __name__ == '__main__':
    from core.loader import load_train_dataset, load_test_dataset
    print('Loading data for iterative induction test suite...')
    sandbox_iterative_induction(
        train_data=load_train_dataset(),
        test_data=load_test_dataset(),
    )
