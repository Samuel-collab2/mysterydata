from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
from sklearn.utils._testing import ignore_warnings
from core.model_base import BaseModel


class StaticRegression(BaseModel):

    def __init__(self, value):
        self._value = value

    def fit(self, train_features, train_labels):
        return

    def predict(self, test_features):
        return [self._value] * len(test_features)

    def score(self, test_features, test_labels):
        return mean_absolute_error(test_labels, self.predict(test_features))


def train_static_regression(train_features, train_label, value=1e-8):
    model = StaticRegression(value=value)
    model.fit(train_features, train_label)
    return model

def train_linear_regression(train_features, train_label):
    model = LinearRegression()
    model.fit(train_features, train_label)
    return model

def train_polynomial_regression(train_features, train_label, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(train_features, train_label)
    return model

@ignore_warnings(category=ConvergenceWarning)
def train_lasso_regression(train_features, train_label, alpha):
    model = Lasso(alpha=alpha)
    model.fit(train_features, train_label)
    return model

@ignore_warnings(category=ConvergenceWarning)
def train_ridge_regression(train_features, train_label, alpha):
    model = Ridge(alpha=alpha)
    model.fit(train_features, train_label)
    return model
