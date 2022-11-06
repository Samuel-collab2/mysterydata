from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from core.model_induction import NullBinaryClassifier


class NeuralNetworkClassifier(NullBinaryClassifier):
    """
    Basic neural network binary classification model.
    """

    HIDDEN_LAYER_SIZE = 16

    @classmethod
    def _compile_model(cls, train_features):
        model = Sequential()
        model.add(Dense(cls.HIDDEN_LAYER_SIZE, activation='relu', input_dim=len(train_features.columns)))
        model.add(Dense(cls.HIDDEN_LAYER_SIZE, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
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
            batch_size=self._batch_size,
            verbose=0)

    def predict(self, test_features):
        return [v >= 0.5
            for vs in self._model.predict(test_features, verbose=0)
                for v in vs]
