from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def train_linear_regression(train_features, train_label):
    model = LinearRegression()
    model.fit(train_features, train_label)
    return model

def train_polynomial_regression(train_features, train_label, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(train_features, train_label)
    return model
