import pickle
import pandas as pd

from core.loader import load_train_dataset, load_determining_dataset
from core.model_set import train_model_from_model_set
from core.preprocessing import get_categorical_columns, separate_features_label, \
    expand_dataset_deterministic, create_augmented_features, convert_label_boolean, \
    split_claims_accept_reject
from core.constants import DATASET_LABEL_NAME, MODEL_PATH
from core.constants_feature_set import SIGNIFICANT_AUGMENTED_COLUMNS
from core.constants_submission import COMPETITION_MODEL_SET


def handle_model_export():
    fit_and_export_model(load_train_dataset(), COMPETITION_MODEL_SET)

def fit_and_export_model(train_dataset, model_set):
    train_features, train_labels = separate_features_label(train_dataset, DATASET_LABEL_NAME)
    categorical_columns = get_categorical_columns(train_dataset)
    categorical_train_features = train_features.loc[:, categorical_columns]
    expanded_train_features = expand_dataset_deterministic(
        train_features,
        load_determining_dataset(),
        categorical_columns
    )
    augmented_train_features = create_augmented_features(
        train_features,
        SIGNIFICANT_AUGMENTED_COLUMNS
    )

    induction_train_features = pd.concat((
        expanded_train_features,
        augmented_train_features,
        categorical_train_features,
    ), axis='columns')
    induction_train_labels = convert_label_boolean(train_labels)
    induction_data = (induction_train_features, induction_train_labels,
        pd.DataFrame(columns=induction_train_features.columns),
        pd.DataFrame(columns=induction_train_features.columns))

    accept_data, _ = split_claims_accept_reject(induction_train_features, train_labels)
    accept_train_features, accept_train_label = accept_data
    regression_data = (accept_train_features, accept_train_label,
        pd.DataFrame(columns=induction_train_features.columns),
        pd.DataFrame(columns=induction_train_features.columns))

    model = train_model_from_model_set(model_set, induction_data, regression_data)
    export_model(model)

def import_model_and_predict(test_dataset):
    model = import_model()
    # create predictions and write to csv

def export_model(model):
    with open(MODEL_PATH, mode='wb') as model_file:
        pickle.dump(model, model_file)
        print(f'Wrote fitted model to {MODEL_PATH}')

def import_model():
    with open(MODEL_PATH, mode='rb') as model_file:
        model = pickle.load(model_file)
        print(f'Read fitted model from {MODEL_PATH}')

    return model
