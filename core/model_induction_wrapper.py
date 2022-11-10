import pandas as pd
from core.model_base import BaseModel


def _null_predicate(_):
    return False


class ModelInductionWrapper(BaseModel):
    """
    Wraps a binary classification model with support for accept/reject overrides.
    """

    def __init__(self, model,
                 predicate_accept=_null_predicate,
                 predicate_reject=_null_predicate):
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

    def _handle_predicate(self, predicate, row):
        # enables accessing nonexistent features in an entry
        # this is important for brevity when working with feature subsets
        try:
            return predicate(row)
        except KeyError:
            return False

    def fit(self, train_features, train_labels):
        return self._model.fit(train_features, train_labels)

    def predict(self, test_features):
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

        pred_labels = (self._model.predict(rows_unsure.loc[:, test_features.columns])
            if not rows_unsure.empty
            else [])

        rows_reject['label'] = False
        rows_accept['label'] = True
        rows_unsure['label'] = pred_labels

        pred_rows = pd.concat((rows_reject, rows_accept, rows_unsure))
        pred_rows.sort_values(by='index')
        return pred_rows.loc[:, 'label']

    def score(self, test_features, test_labels):
        return self._model.score(test_features, test_labels)
