from core.constants_feature_set import ALL_EXPANDED_FEATURES
from core.preprocessing import balance_binary_dataset


def modify_model(train_model, **kwargs):
    """
    Helper that allows args to be passed to a model without specifying the full anonymous function.
    :param train_model: The train model function
    :param kwargs: The constructor arguments for the model
    :return: Augmented train model function
    """
    return lambda features, label: train_model(features, label, **kwargs)

def modifier_filter_columns(columns):
    """
    Modifier to filter data to only contain the given columns.
    :param columns: The columns to include
    :return: Modifier function
    """
    def modify(train_features, train_label, test_features, evaluation_features):
        return train_features.loc[:, columns], train_label, test_features.loc[:, columns], evaluation_features.loc[:, columns]

    return modify

def modifier_balance_binary_data(skew_true=1, skew_false=1):
    """
    Modifier to train a model with skewed true vs false binary labels.
    :param skew_true: The skew true value
    :param skew_false: The skew false value
    :return: Modifier function
    """
    def modify(train_features, train_label, test_features, evaluation_features):
        train_features, train_label = balance_binary_dataset(train_features, train_label, skew_true, skew_false)
        return train_features, train_label, test_features, evaluation_features

    return modify

def modifier_filter_expanded():
    return modifier_filter_columns(ALL_EXPANDED_FEATURES)
