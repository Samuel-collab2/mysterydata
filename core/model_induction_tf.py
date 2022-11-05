from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD

from core.model_induction import NullDecisionTreeInduction


class NeuralNetworkClassifier(NullDecisionTreeInduction):
    """
    Basic neural network binary classification model.
    """

    @staticmethod
    def _compile_model(train_features):
        model = Sequential()
        model.add(Dense(8, input_shape=(len(train_features.columns),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=0.001),
                      metrics=['accuracy',
                               Precision(),
                               Recall()])
        return model

    def __init__(self, epochs, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._model = None

    def fit(self, train_features, train_labels):
        self._model = self._compile_model(train_features)
        self._model.fit(train_features, train_labels,
            epochs=self._epochs,
            batch_size=self._batch_size)

    def predict(self, test_features):
        return [v >= 0.5
            for vs in self._model.predict(test_features)
                for v in vs]
