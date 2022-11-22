from sklearn.metrics import mean_absolute_error

from core.constants_submission import COMPETITION_MODEL_SET
from core.loader import load_test_dataset, load_train_dataset, load_competition_dataset
from core.model_exporter import import_model
from core.model_set import get_model_set_data, get_model_set_predictions, modify_model_set_train_data
from core.predict import output_predictions

OUTPUT_FILE_NAME = 'predictedclaimamount'

def get_predictions(model, model_set, submission_data):
    induction_data, regression_data, evaluation_label = submission_data
    induction_data, regression_data = modify_model_set_train_data(
        model_set,
        induction_data,
        regression_data,
    )

    _, _, induction_features, induction_evaluation_features = induction_data
    _, _, regression_features, regression_evaluation_features = regression_data

    evaluation_predictions = model.predict(
        induction_evaluation_features,
        regression_evaluation_features
    )

    test_predictions = model.predict(
        induction_features,
        regression_features,
    )

    return evaluation_predictions, evaluation_label, test_predictions

def main():
    competition_dataset = load_competition_dataset()
    train_dataset, model = import_model()

    model_set_data = get_model_set_data(train_dataset, competition_dataset)
    evaluation_predictions, evaluation_label, competition_predictions = get_predictions(
        model,
        COMPETITION_MODEL_SET,
        model_set_data
    )

    training_mae = mean_absolute_error(
        evaluation_label,
        evaluation_predictions
    )

    print(f'Training MAE: {training_mae}')

    output_predictions(competition_predictions, OUTPUT_FILE_NAME)


if __name__ == '__main__':
    main()
