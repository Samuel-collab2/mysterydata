import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_mae(label, predictions):
    return np.mean(np.abs(label - predictions))

def handle_linear_regression(train_features, train_label, samples):
    model = LinearRegression()
    model.fit(train_features, train_label)
    return model.predict(samples)
