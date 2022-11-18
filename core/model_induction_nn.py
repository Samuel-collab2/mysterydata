from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.metrics import AUC
from keras import backend as K

from core.model_induction import NullBinaryClassifier


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class NeuralNetworkClassifier(NullBinaryClassifier):
    """
    Basic neural network binary classification model.
    """

    def __init__(self, epochs, batch_size,
                 hidden_layer_size=16,
                 optimizer='adam',
                 activation='relu',
                 verbose=0,
                 class_weight=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._hidden_layer_size = hidden_layer_size
        self._optimizer = optimizer
        self._activation = activation
        self._verbose = verbose
        self._class_weight = class_weight
        self._model = None

    def _compile_model(self, train_features):
        model = Sequential()
        model.add(Dense(self._hidden_layer_size,
            activation=self._activation,
            kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.1))
        model.add(Dense(self._hidden_layer_size,
            activation=self._activation,
            kernel_regularizer=l2(0.001),
            input_dim=len(train_features.columns)))
        model.add(Dropout(0.1))
        model.add(Dense(1,
            activation='sigmoid',
            kernel_regularizer=l2(0.001)))
        model.compile(loss='binary_crossentropy',
                      optimizer=self._optimizer,
                      metrics=[
                          'sparse_categorical_accuracy',
                          'binary_accuracy',
                          f1_m,
                          AUC(),
                      ])
        return model

    def fit(self, train_features, train_labels, *args, **kwargs):
        self._model = self._compile_model(train_features)
        self._model.fit(train_features, train_labels,
            epochs=self._epochs,
            batch_size=self._batch_size,
            class_weight=self._class_weight,
            verbose=self._verbose,
            *args, **kwargs)

    def predict(self, test_features):
        return [v >= 0.5
            for vs in self._model.predict(test_features, verbose=0)
                for v in vs]

def train_network_classifier(train_features, train_label, **kwargs):
    model = NeuralNetworkClassifier(**kwargs)
    model.fit(train_features, train_label)
    return model
