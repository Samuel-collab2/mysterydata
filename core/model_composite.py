from os.path import join

import pandas as pd

from core.constants import OUTPUT_DIR
from core.model_induction import train_decision_tree, BinaryDecisionTreeInduction
from core.model_regression import train_linear_regression

class CompositeModel:
    def __init__(self, induction_export_filename):
        self._induction_model = None
        self._regression_model = None
        self._induction_export_filename = induction_export_filename

    def train(self, induction_train_features, induction_train_label, regression_train_features, regression_train_label):
        export_path = join(OUTPUT_DIR, f'{self._induction_export_filename}.json')

        try:
            self._induction_model = BinaryDecisionTreeInduction.load_from_path(export_path)
        except FileNotFoundError:
            self._induction_model = train_decision_tree(induction_train_features, induction_train_label)
            self._induction_model.dump(export_path)

        self._regression_model = train_linear_regression(regression_train_features, regression_train_label)

    def predict(self, induction_test_features, regression_test_features):
        induction_predictions = self._induction_model.predict(induction_test_features)

        return [
            self._regression_model.predict(pd.DataFrame([regression_test_features.iloc[index]]))[0]
            if prediction
            else 0
            for index, prediction
            in enumerate(induction_predictions)
        ]

def train_composite(induction_train_features, induction_train_label, regression_train_features, regression_train_label, export_filename):
    model = CompositeModel(export_filename)
    model.train(induction_train_features, induction_train_label, regression_train_features, regression_train_label)
    return model
