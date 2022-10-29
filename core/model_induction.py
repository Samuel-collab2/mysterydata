from math import log2
import pandas as pd


class BinaryDecisionTree:

    @staticmethod
    def _find_root_entropy(label):
        num_labels = len(label)
        num_true = sum(label)
        num_false = num_labels - num_true
        if num_true == 0 or num_false == 0:
            return 0

        p_true = num_true / num_labels
        p_false = num_false / num_labels
        return -(p_true * log2(p_true)
            + p_false * log2(p_false))

    @staticmethod
    def _find_feature_values(samples, feature):
        return list(set(samples.loc[:, feature]))


    def __init__(self):
        self._tree = None
        self._dataset_num_features = 0
        self._dataset_label_name = None

    def _find_entropy(self, samples, feature):
        entropy = 0
        for feature_value in self._find_feature_values(samples, feature):
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

            feature_entropy = -(p_true * log2(p_true)
                + p_false * log2(p_false))
            feature_weight = num_feature_value_samples / num_samples

            entropy += feature_weight * feature_entropy

        return entropy

    def _find_information_gain(self, samples, feature, entropy):
        return entropy - self._find_entropy(samples, feature)

    def _induce(self, samples):
        ply = self._dataset_num_features - len(samples.columns) + 2

        if len(samples.columns) == 1:
            return  # STOPPAGE CRITERIA: label is only column in dataset

        root_entropy = self._find_root_entropy(samples.loc[:, self._dataset_label_name])
        if root_entropy == 0:
            # STOPPAGE CRITERIA: perfect classification: all samples in dataset have the same label value
            return samples.loc[:, self._dataset_label_name].any()

        feature_gains = {feature: self._find_information_gain(
            samples=samples,
            feature=feature,
            entropy=root_entropy,
        ) for feature in samples.columns
            if feature != self._dataset_label_name}

        feature_gain_values = list(feature_gains.values())
        max_feature_gain = max(feature_gain_values)
        max_feature_index = feature_gain_values.index(max_feature_gain)
        max_feature = list(feature_gains.keys())[max_feature_index]

        other_features = [feature for feature in samples.columns if feature != max_feature]
        for feature_value in self._find_feature_values(samples, max_feature):
            child_samples = samples[samples[max_feature] == feature_value].loc[:, other_features]
            print(f'Induce {max_feature}'
                f' through {feature_value}'
                f' at ply {ply + 1}'
                f' ({len(child_samples)} samples)')
            self._induce(child_samples)

    def train(self, train_features, train_label):
        self._dataset_num_features = len(train_features.columns)
        self._dataset_label_name = train_label.name
        train_samples = pd.concat((train_features, train_label), axis=1)
        self._induce(train_samples)

    def predict(self, test_features):
        return [False
            for i, x in test_features.iterrows()]


def train_decision_tree(train_features, train_label):
    model = BinaryDecisionTree()
    model.train(train_features, train_label)
    return model
