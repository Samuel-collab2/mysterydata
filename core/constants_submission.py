from core.constants import SIGNIFICANT_BINARY_LABEL_COLUMNS, SIGNIFICANT_RIDGE_COLUMNS
from core.model_induction import train_classifier_tree, train_decision_tree
from core.model_induction_nn import train_classifier_network
from core.model_regression import train_linear_regression
from core.modifiers import modifier_filter_columns, modify_model, modifier_balance_binary_data

SUBMISSION1_RIDGE_FEATURE_SET_COUNTS = [1, 3, 7, 5, 10]
SUBMISSION1_PROPAGATION_FEATURE_SET_COUNTS = [3, 5, 10, 15, 20]

# Model sets to use for submission 2
SUBMISSION2_MODEL_SETS = [
    (
        'All feature regression test (Jonathan 1)',
        train_decision_tree, [],
        train_linear_regression, []
    ),
    (
        '10-feature regression test (Jonathan 2)',
        train_decision_tree, [],
        train_linear_regression, [
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10]),
        ]
    ),
    (
        'Respective top 3',
        train_classifier_tree, [
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS[:3]),
        ], train_linear_regression, [
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:3]),
        ]
    ),
    (
        'Complex model setup example',
        modify_model(train_classifier_tree, max_depth=10), [
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS[:10]),
            modifier_balance_binary_data(skew_false=8)
        ], train_linear_regression, [
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10]),
        ]
    ),
    (
        'Neural network classifier',
        modify_model(train_classifier_network, epochs=50, batch_size=100), [],
        train_linear_regression, []
    ),
]
