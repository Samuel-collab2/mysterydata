from core.constants import SIGNIFICANT_BINARY_LABEL_COLUMNS, SIGNIFICANT_RIDGE_COLUMNS, \
    SIGNIFICANT_FORWARD_STEPWISE_COLUMNS
from core.model_induction import train_random_forest, train_decision_tree
from core.model_induction_nn import train_network_classifier
from core.model_regression import train_linear_regression, train_polynomial_regression
from core.model_set import ModelSet
from core.model_set_modifiers import modifier_filter_columns, modify_model, modifier_balance_binary_data

SUBMISSION1_RIDGE_FEATURE_SET_COUNTS = [1, 3, 7, 5, 10]
SUBMISSION1_PROPAGATION_FEATURE_SET_COUNTS = [3, 5, 10, 15, 20]

# Model sets to use for submission 2
SUBMISSION2_MODEL_SETS = [
    ModelSet(
        name='Linear regression: All features',
        train_induction_model=train_decision_tree,
    ),
    ModelSet(
        name='Linear regression: 10 ridge features',
        train_induction_model=train_decision_tree,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10]),
        ]
    ),
    ModelSet(
        name='Polynomial regression: 3 ridge features, degree 9',
        train_induction_model=train_decision_tree,
        train_regression_model=modify_model(train_polynomial_regression, degree=9),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:3]),
        ]
    ),
    ModelSet(
        name='Polynomial regression: 5 ridge features, degree 6',
        train_induction_model=train_decision_tree,
        train_regression_model=modify_model(train_polynomial_regression, degree=6),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:5]),
        ]
    ),
    ModelSet(
        name='Random forest: All binary-label ridge features, 70 estimators, 80 depth, rejection-skew 8',
        train_induction_model=modify_model(train_random_forest, n_estimators=70, max_depth=80),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=train_linear_regression,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10])
        ],
    ),
    ModelSet(
        name='Random forest: All binary-label ridge features, 50 estimators, ∞ depth, rejection-skew 8',
        train_induction_model=modify_model(train_random_forest, n_estimators=50, max_depth=None),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=train_linear_regression,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10])
        ],
    ),
    ModelSet(
        name='Random forest: All binary-label ridge features, 30 estimators, 60 depth, rejection-skew 8',
        train_induction_model=modify_model(train_random_forest, n_estimators=30, max_depth=60),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=train_linear_regression,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10])
        ],
    ),
    ModelSet(
        name='Random forest: All propagation features, 50 estimators, 40 depth, rejection-skew 8',
        train_induction_model=modify_model(train_random_forest, n_estimators=50, max_depth=40),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_FORWARD_STEPWISE_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=train_linear_regression,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10])
        ],
    ),
    ModelSet(
        name='Neural network classifier: All propagation features, 50 epochs, 100 batch size, relu activation, rejection-skew 1',
        train_induction_model=modify_model(train_network_classifier, epochs=50, batch_size=100, activation='relu'),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_FORWARD_STEPWISE_COLUMNS),
            modifier_balance_binary_data(),
        ],
        train_regression_model=train_linear_regression,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10])
        ],
    ),
    ModelSet(
        name='Neural network classifier: All binary-label ridge features, 50 epochs, 100 batch size, tanh activation, rejection-skew 2',
        train_induction_model=modify_model(train_network_classifier, epochs=50, batch_size=100, activation='tanh'),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=2),
        ],
        train_regression_model=train_linear_regression,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10])
        ],
    ),
]
