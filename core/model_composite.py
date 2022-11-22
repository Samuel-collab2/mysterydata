import pandas as pd
from core.model_induction import train_random_forest
from core.model_regression import train_linear_regression

class CompositeModel:
    def __init__(self, train_induction_model, train_regression_model, proba_threshold=0.5):
        self._train_induction_model = train_induction_model
        self._train_regression_model = train_regression_model
        self._induction_model = None
        self._regression_model = None
        self._proba_threshold = proba_threshold  # required probability to make prediction

    def train(self, induction_train_features, induction_train_label, regression_train_features, regression_train_label):
        self._induction_model = self._train_induction_model(induction_train_features, induction_train_label)
        self._regression_model = self._train_regression_model(regression_train_features, regression_train_label)

        # HACK: unset lambdas for pickling
        self._train_induction_model = None
        self._train_regression_model = None

    def predict(self, induction_test_features, regression_test_features):
        induction_proba = self._induction_model.predict_proba(induction_test_features)[:, 1]
        return [
            self._regression_model.predict(pd.DataFrame([regression_test_features.iloc[index]]))[0]
                if proba >= self._proba_threshold
                else (1e-8 if proba >= 0.5 else 0)
                    for index, proba in enumerate(induction_proba)
        ]

def train_composite(
    induction_train_features, induction_train_label,
    regression_train_features, regression_train_label,
    train_induction_model, train_regression_model,
    proba_threshold=0.5,
):
    model = CompositeModel(train_induction_model, train_regression_model,
        proba_threshold=proba_threshold)
    model.train(
        induction_train_features, induction_train_label,
        regression_train_features, regression_train_label,
    )
    return model
