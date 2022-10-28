from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


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
