import pandas as pd
import numpy as np
from core.model_base import BaseModel


def _null_predicate(_):
    return False


class ModelInductionWrapper(BaseModel):
    """
    Wraps a binary classification model with support for accept/reject overrides.
    """

    def __init__(self, model,
                 predicate_accept=_null_predicate,
                 predicate_reject=_null_predicate,
                 model_columns=None):
        """
        Initializes a binary classification model wrapper.
        :param model: the model to wrap
        :param predicate_accept: a function row->bool
                                 return true to always accept regardless of classifier results
        :param predicate_reject: a function row->bool;
                                 return true to always reject regardless of classifier results
        """
        self._model = model
        self._predicate_accept = predicate_accept
        self._predicate_reject = predicate_reject
        self._model_columns = model_columns

    def get_model_columns(self, features):
        return features.columns if self._model_columns is None else self._model_columns

    def _handle_predicate(self, predicate, row):
        # enables accessing nonexistent features in an entry
        # this is important for brevity when working with feature subsets
        try:
            return predicate(row)
        except KeyError:
            return False

    def fit(self, train_features, train_labels):
        return self._model.fit(train_features.loc[:, self.get_model_columns(train_features)], train_labels)

    def predict(self, test_features):
        return self._model.predict_proba(test_features)[:, 1] > 0.5

    def predict_proba(self, test_features):
        test_columns = self.get_model_columns(test_features)

        pred_features = test_features.copy()
        pred_accepts, pred_rejects = zip(*((
            self._handle_predicate(self._predicate_accept, row),
            self._handle_predicate(self._predicate_reject, row)
        ) for i, row in pred_features.iterrows()))

        pred_features['accept'] = pred_accepts
        pred_features['reject'] = pred_rejects
        pred_features['index'] = range(len(pred_features))

        # reject takes precedence over accept (tie-break)
        rows_reject = pred_features[pred_features['reject'] == True].copy()
        rows_accept = pred_features[
            (pred_features['accept'] == True)
            & (pred_features['reject'] == False)
        ].copy()
        rows_unsure = pred_features[
            (pred_features['accept'] == False)
            & (pred_features['reject'] == False)
        ].copy()

        pred_proba = (self._model.predict_proba(rows_unsure.loc[:, test_columns])[:, 1]
            if not rows_unsure.empty
            else [])

        rows_reject['label'] = 0
        rows_accept['label'] = 1
        rows_unsure['label'] = pred_proba

        pred_rows = pd.concat((rows_reject, rows_accept, rows_unsure))
        pred_rows.sort_values(by='index')
        return np.array([
            [0] * len(pred_rows),  # HACK: should be inverse of label values (usually unused)
            pred_rows.loc[:, 'label'].values,
        ]).T

    def score(self, test_features, test_labels):
        return self._model.score(test_features, test_labels)

def train_wrapped_induction(train_features, train_label, model, predicate_accept, predicate_reject):
    model = ModelInductionWrapper(model, predicate_accept, predicate_reject)
    model.fit(train_features, train_label)
    return model

def predicate_accept_brandon(claim):
    return (claim['feature11'] == 5
        or claim['feature9'] == 0
        or claim['feature13'] == 4
        or claim['feature14'] == 3
        or claim['feature18'] == 1)

def predicate_reject_brandon(claim):
    return claim['feature7'] == 3
