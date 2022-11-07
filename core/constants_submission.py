from core.constants import SIGNIFICANT_BINARY_LABEL_COLUMNS, SIGNIFICANT_RIDGE_COLUMNS
from core.model_induction import train_random_forest, train_decision_tree
from core.model_induction_nn import train_network_classifier
from core.model_regression import train_linear_regression
from core.model_set import ModelSet
from core.model_set_modifiers import modifier_filter_columns, modify_model, modifier_balance_binary_data

SUBMISSION1_RIDGE_FEATURE_SET_COUNTS = [1, 3, 7, 5, 10]
SUBMISSION1_PROPAGATION_FEATURE_SET_COUNTS = [3, 5, 10, 15, 20]

# Model sets to use for submission 2
SUBMISSION2_MODEL_SETS = [
    ModelSet(
        name='All feature regression test (Jonathan 1)',
        train_induction_model=train_decision_tree,
    ),
    ModelSet(
        name='10-feature regression test (Jonathan 2)',
        train_induction_model=train_decision_tree,
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10]),
        ]
    ),
    ModelSet(
        name='Respective top 3',
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS[:3]),
        ],
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:3]),
        ]
    ),
    ModelSet(
        name='Complex model setup example',
        train_induction_model=modify_model(train_random_forest, max_depth=10),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS[:10]),
            modifier_balance_binary_data(skew_false=8)
        ],
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:10]),
        ]
    ),
    ModelSet(
        name='Neural network classifier',
        train_induction_model=modify_model(train_network_classifier, epochs=50, batch_size=100),
    ),
]
