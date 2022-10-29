from math import log2
import pandas as pd


class BinaryDecisionTree:

    def __init__(self):
        self._tree = None
        self._dataset_label_name = None

    def _find_root_entropy(self, label):
        num_labels = len(label)
        num_true = sum(label)
        num_false = num_labels - num_true

        p_true = num_true / num_labels
        p_false = num_false / num_labels

        return -(p_true * log2(p_true)
            + p_false * log2(p_false))

    def _find_entropy(self, samples, feature):
        feature_values = list(set(samples.loc[:, feature]))

        entropy = 0
        for feature_value in feature_values:
            print(f'\n-- {feature}={feature_value}')

            num_samples = len(samples)
            feature_value_samples = samples[samples[feature] == feature_value]
            num_feature_value_samples = len(feature_value_samples)
            if num_feature_value_samples == 0:
                continue

            num_true_samples = len(feature_value_samples[
                feature_value_samples[self._dataset_label_name] == True])
            num_false_samples = len(feature_value_samples[
                feature_value_samples[self._dataset_label_name] == False])
            if num_true_samples == 0 or num_false_samples == 0:
                continue

            p_true = num_true_samples / num_feature_value_samples
            p_false = num_false_samples / num_feature_value_samples
            print(f'p_true={num_true_samples}/{num_feature_value_samples}')
            print(f'p_false={num_false_samples}/{num_feature_value_samples}')

            feature_entropy = -(p_true * log2(p_true)
                + p_false * log2(p_false))
            feature_weight = num_feature_value_samples / num_samples

            print(f'entropy={feature_entropy:.4f}'
                f'\nweight={num_feature_value_samples}/{num_samples}')
            entropy += feature_weight * feature_entropy

        return entropy

    def _find_information_gain(self, samples, feature, entropy):
        return entropy - self._find_entropy(samples, feature)

    def train(self, train_features, train_label):
        self._dataset_label_name = train_label.name

        root_entropy = self._find_root_entropy(train_label)
        print(f'Root entropy: {root_entropy:.4f}')

        train_samples = pd.concat((train_features, train_label), axis=1)
        feature_gains = {feature: self._find_information_gain(
            samples=train_samples,
            feature=feature,
            entropy=root_entropy,
        ) for feature in train_features.columns}

        for feature, feature_gain in feature_gains.items():
            print(f'{feature} information gain = {feature_gain:.4f}')

    def predict(self, test_features):
        return [False for i, x in test_features.iterrows()]


def train_decision_tree(train_features, train_label):
    model = BinaryDecisionTree()
    model.train(train_features, train_label)
    return model
