import pandas as pd


class DecisionTree:
    def train(self, train_features, train_label):
        pass

    def predict(self, test_features):
        return [False for i, x in test_features.iterrows()]


def train_decision_tree(train_features, train_label):
    model = DecisionTree()
    model.train(train_features, train_label)
    return model
