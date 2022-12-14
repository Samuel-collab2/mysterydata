from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from core.model_induction import NullBinaryClassifier


class NeuralNetworkClassifier(NullBinaryClassifier):
    """
    Basic neural network binary classification model.
    """

    def __init__(self, epochs, batch_size,
                 hidden_layer_size=16,
                 optimizer='adam',
                 activation='relu',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._hidden_layer_size = hidden_layer_size
        self._optimizer = optimizer
        self._activation = activation
        self._model = None

    def _compile_model(self, train_features):
        model = Sequential()
        model.add(Dense(self._hidden_layer_size, activation=self._activation,
            input_dim=len(train_features.columns)))
        model.add(Dense(self._hidden_layer_size, activation=self._activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=self._optimizer,
                      metrics=['accuracy'])
        return model

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

def train_network_classifier(train_features, train_label, **kwargs):
    model = NeuralNetworkClassifier(**kwargs)
    model.fit(train_features, train_label)
    return model
