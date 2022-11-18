from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import TomekLinks, SMOTETomek

from core.constants_feature_set import SIGNIFICANT_RIDGE_COLUMNS, SIGNIFICANT_BINARY_LABEL_COLUMNS, \
    SIGNIFICANT_FORWARD_STEPWISE_COLUMNS, SIGNIFICANT_AUGMENTED_COLUMNS
from core.model_induction import train_random_forest, train_decision_tree, train_svc
from core.model_induction_nn import train_network_classifier
from core.model_induction_wrapper import train_wrapped_induction, predicate_accept_brandon, predicate_reject_brandon
from core.model_regression import train_static_regression, train_linear_regression, train_polynomial_regression
from core.model_set import ModelSet
from core.model_set_modifiers import modifier_filter_columns, modify_model, \
    modifier_balance_binary_data, modifier_resample

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


SUBMISSION3_MODEL_SETS = [
    ModelSet(
        name='Test random forest with static regression model',
        train_induction_model=modify_model(train_random_forest, n_estimators=50, max_depth=None),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=train_static_regression,
    ),
    ModelSet(
        name='Best submission 2 induction and regression',
        train_induction_model=modify_model(train_random_forest, n_estimators=50, max_depth=None),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=modify_model(train_polynomial_regression, degree=9),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:3]),
        ]
    ),
    ModelSet(
        name='Best submission 2 induction and regression, augmented induction features',
        train_induction_model=modify_model(train_random_forest, n_estimators=50, max_depth=None),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=modify_model(train_polynomial_regression, degree=9),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:3]),
        ]
    ),
    ModelSet(
        name='Best submission induction, degree 2 all significant stepwise feature regression',
        train_induction_model=modify_model(train_random_forest, n_estimators=50, max_depth=None),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=8),
        ],
        train_regression_model=modify_model(train_polynomial_regression, degree=2),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_FORWARD_STEPWISE_COLUMNS),
        ]
    ),
    ModelSet(
        name='Wrapped augmentation, skew 6, static regression',
        train_induction_model=modify_model(
            train_wrapped_induction,
            model=RandomForestClassifier(n_estimators=50, max_depth=None),
            predicate_accept=predicate_accept_brandon,
            predicate_reject=predicate_reject_brandon,
        ),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_balance_binary_data(skew_false=6),
        ],
        train_regression_model=train_static_regression,
    ),
    ModelSet(
        name='Wrapped augmentation, skew 6, 9-degree polynomial regression',
        train_induction_model=modify_model(
            train_wrapped_induction,
            model=RandomForestClassifier(n_estimators=50, max_depth=None),
            predicate_accept=predicate_accept_brandon,
            predicate_reject=predicate_reject_brandon,
        ),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_balance_binary_data(skew_false=6),
        ],
        train_regression_model=modify_model(train_polynomial_regression, degree=9),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:3]),
        ]
    ),
    ModelSet(
        name='Augmented random forest, skew 6, static regression',
        train_induction_model=modify_model(train_random_forest, n_estimators=90, max_depth=None),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_balance_binary_data(skew_false=6),
        ],
        train_regression_model=train_static_regression,
    ),
    ModelSet(
        name='Augmented random forest, skew 6, 9-degree polynomial regression',
        train_induction_model=modify_model(train_random_forest, n_estimators=90, max_depth=None),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_balance_binary_data(skew_false=6),
        ],
        train_regression_model=modify_model(train_polynomial_regression, degree=9),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS[:3]),
        ]
    ),
]

SUBMISSION4_MODEL_SETS = [
    ModelSet(
        name='\"Golden Child\" (balanced class weights, no bootstrapping)',
        train_induction_model=modify_model(
            train_wrapped_induction,
            model=RandomForestClassifier(
                n_estimators=50,
                class_weight='balanced',
                bootstrap=False
            ),
            predicate_accept=predicate_accept_brandon,
            predicate_reject=predicate_reject_brandon,
        ),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
        ],
        train_regression_model=train_static_regression,
    ),
    ModelSet(
        name='Balanced class weights, no bootstrapping, Tomek links removed',
        train_induction_model=modify_model(
            train_wrapped_induction,
            model=RandomForestClassifier(
                n_estimators=50,
                class_weight='balanced',
                bootstrap=False
            ),
            predicate_accept=predicate_accept_brandon,
            predicate_reject=predicate_reject_brandon,
        ),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_resample(TomekLinks()),
        ],
        train_regression_model=train_static_regression,
    ),
    ModelSet(
        name='Balanced class weights, no bootstrapping, Tomek links removed, SMOTE oversampling',
        train_induction_model=modify_model(
            train_wrapped_induction,
            model=RandomForestClassifier(
                n_estimators=50,
                class_weight='balanced',
                bootstrap=False
            ),
            predicate_accept=predicate_accept_brandon,
            predicate_reject=predicate_reject_brandon,
        ),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_resample(SMOTETomek()),
        ],
        train_regression_model=train_static_regression,
    ),
    ModelSet(
        name='Submission 3 winner, balanced class weights, all ridge features, degree 2',
        train_induction_model=modify_model(
            train_wrapped_induction,
            model=RandomForestClassifier(n_estimators=50, class_weight='balanced'),
            predicate_accept=predicate_accept_brandon,
            predicate_reject=predicate_reject_brandon,
        ),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_AUGMENTED_COLUMNS),
            modifier_balance_binary_data(skew_false=6),
        ],
        train_regression_model=modify_model(train_polynomial_regression, degree=2),
        regression_modifiers=[
            modifier_filter_columns(SIGNIFICANT_RIDGE_COLUMNS),
        ]
    ),
    # Leave SVC models last, they take a bit
    ModelSet(
        name='SVC static',
        train_induction_model=modify_model(train_svc, penalty=50),
        induction_modifiers=[
            modifier_filter_columns(SIGNIFICANT_BINARY_LABEL_COLUMNS),
            modifier_balance_binary_data(skew_false=12),
        ],
        train_regression_model=train_static_regression,
    ),
]
