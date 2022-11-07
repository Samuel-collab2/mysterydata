from core.preprocessing import balance_binary_dataset


def modify_model(model, **kwargs):
    return lambda features, label: model(features, label, **kwargs)

def modifier_filter_columns(columns):
    def modify(train_features, train_label, test_features, evaluation_features):
        return train_features.loc[:, columns], train_label, test_features.loc[:, columns], evaluation_features.loc[:, columns]

    return modify

def modifier_balance_binary_data(skew_true=1, skew_false=1):
    def modify(train_features, train_label, test_features, evaluation_features):
        train_features, train_label = balance_binary_dataset(train_features, train_label, skew_true, skew_false)
        return train_features, train_label, test_features, evaluation_features

    return modify
