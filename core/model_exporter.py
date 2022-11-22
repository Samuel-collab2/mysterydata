import pickle

from core.loader import load_train_dataset
from core.model_set import train_model_from_model_set, get_model_set_train_data
from core.constants import MODEL_PATH
from core.constants_submission import COMPETITION_MODEL_SET


def handle_model_export():
    fit_and_export_model(load_train_dataset(), COMPETITION_MODEL_SET)

def fit_and_export_model(train_dataset, model_set):
    induction_data, regression_data = get_model_set_train_data(train_dataset)
    model = train_model_from_model_set(model_set, induction_data, regression_data)
    export_model(train_dataset, model)

def export_model(train_dataset, model):
    with open(MODEL_PATH, mode='wb') as model_file:
        pickle.dump((train_dataset, model), model_file)
        print(f'Wrote fitted model to {MODEL_PATH}')

def import_model():
    with open(MODEL_PATH, mode='rb') as model_file:
        train_dataset, model = pickle.load(model_file)
        print(f'Read fitted model from {MODEL_PATH}')

    return train_dataset, model
