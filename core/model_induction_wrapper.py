import pandas as pd
from core.model_base import BaseModel


class ModelInductionWrapper(BaseModel):

    def __init__(self, model, predicate):
        self._model = model
        self._predicate = predicate

    def fit(self, train_features, train_labels):
        return self._model.fit(train_features, train_labels)

    def predict(self, test_features):
        pred_features = test_features.copy()
        pred_features['accept'] = [self._predicate(row)
            for i, row in pred_features.iterrows()]
        pred_features['index'] = range(len(pred_features))
        bad_rows = pred_features[pred_features['accept'] == False]
        good_rows = pred_features[pred_features['accept'] == True]

        pred_labels = (self._model.predict(good_rows.loc[:, test_features.columns])
            if not good_rows.empty
            else [])

        bad_rows['label'] = 0
        good_rows['label'] = pred_labels

        pred_rows = pd.concat((bad_rows, good_rows))
        pred_rows.sort_values(by='index')
        return pred_rows.loc[:, 'label']

    def score(self, test_features, test_labels):
        return self._model.score(test_features, test_labels)
