from math import log2
import pandas as pd


class BinaryDecisionTree:

    @staticmethod
    def _find_root_entropy(label):
        num_labels = len(label)
        num_true = sum(label)
        num_false = num_labels - num_true

        p_true = num_true / num_labels
        p_false = num_false / num_labels

        return -(p_true * log2(p_true)
            + p_false * log2(p_false))

    def train(self, train_features, train_label):
        train_samples = pd.concat((train_features, train_label), axis=1)
        print(f'Root entropy: {self._find_root_entropy(train_label):.4f}')

    def predict(self, test_features):
        return [False for i, x in test_features.iterrows()]


def train_decision_tree(train_features, train_label):
    model = BinaryDecisionTree()
    model.train(train_features, train_label)
    return model
