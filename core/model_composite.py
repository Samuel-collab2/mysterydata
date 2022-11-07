import pandas as pd
from core.model_induction import train_classifier_tree
from core.model_regression import train_linear_regression

class CompositeModel:
    def __init__(self, train_induction_model, train_regression_model):
        self._train_induction_model = train_induction_model
        self._train_regression_model = train_regression_model
        self._induction_model = None
        self._regression_model = None

    def train(self, induction_train_features, induction_train_label, regression_train_features, regression_train_label):
        self._induction_model = self._train_induction_model(induction_train_features, induction_train_label)
        self._regression_model = self._train_regression_model(regression_train_features, regression_train_label)

    def predict(self, induction_test_features, regression_test_features):
        induction_predictions = self._induction_model.predict(induction_test_features)

        return [
            self._regression_model.predict(pd.DataFrame([regression_test_features.iloc[index]]))[0]
            if prediction
            else 0
            for index, prediction
            in enumerate(induction_predictions)
        ]

def train_composite(
    induction_train_features, induction_train_label,
    regression_train_features, regression_train_label,
    train_induction_model, train_regression_model
):
    model = CompositeModel(train_induction_model, train_regression_model)
    model.train(
        induction_train_features, induction_train_label,
        regression_train_features, regression_train_label,
    )
    return model
