from math import log2
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from library.graph import Graph
from core.preprocessing import balance_binary_dataset


class NullBinaryClassifier:

    def fit(self, train_features: pd.DataFrame, train_labels: pd.Series):
        pass

    def predict(self, test_features: pd.DataFrame):
        return [False
            for i, sample in test_features.iterrows()]

    def score(self, features: pd.DataFrame, labels: pd.Series):
        return accuracy_score(self.predict(features), labels)


class CustomBinaryClassifier(NullBinaryClassifier):

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

    @classmethod
    def load_from_path(cls, file_path):
        with open(file_path, mode='r', encoding='utf-8') as file:
            file_buffer = file.read()

        tree_edges = list(map(tuple, json.loads(file_buffer)))
        tree_nodes = list(set([node for edge in tree_edges
            for node in edge[0].split('/')
                if not isinstance(node, bool)]))

        return cls(Graph(
            nodes=sorted(tree_nodes, key=lambda n: -1 if n == 'feature3' else 1),
            edges=tree_edges,
        ))

    def __init__(self, tree=None):
        self._tree = tree
        self._dataset_num_features = 0
        self._dataset_label_name = None

    @property
    def tree(self):
        return self._tree

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

    def _induce(self, samples, path=None):
        path = path or []

        if len(samples.columns) == 1:
            # STOPPAGE CRITERIA: label is only column in dataset
            #                    perform tie break
            num_true_samples = len(samples[samples[self._dataset_label_name] == True])
            num_samples = len(samples)
            return num_true_samples > num_samples // 2

        root_entropy = self._find_root_entropy(samples.loc[:, self._dataset_label_name])
        if root_entropy == 0:
            # STOPPAGE CRITERIA: perfect classification
            #                    all samples in dataset have the same label value
            return bool(samples.loc[:, self._dataset_label_name].any())

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
            child_path = (*path, max_feature)
            child = self._induce(child_samples, path=child_path)
            self._tree.connect(
                node1='/'.join(child_path),
                node2=child,
                data=feature_value)

        return max_feature

    def fit(self, train_features, train_label):
        self._tree = Graph(nodes=train_features.columns)
        self._dataset_num_features = len(train_features.columns)
        self._dataset_label_name = train_label.name

        train_samples = pd.concat((train_features, train_label), axis=1)
        self._induce(train_samples)

    def traverse(self, sample):
        feature = self._tree.nodes[0]
        node = (feature, feature)  # node is column name + column path

        while node is not None:
            feature_name, feature_path = node
            feature_value = sample[feature_name]

            child = next((n2 for n1, n2, edge in self._tree.edges
                if edge == feature_value
                and n1 == feature_path), None)

            if child is None or isinstance(child, bool):
                return child or False  # assume false if no data is provided

            node = (child, f'{feature_path}/{child}')

    def predict(self, test_features):
        return [self.traverse(sample)
            for i, sample in test_features.iterrows()]

    def dump(self, file_path):
        file_buffer = json.dumps(self._tree.edges, separators=(',', ':'))
        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(file_buffer)


def train_custom_tree(train_features, train_label):
    model = CustomBinaryClassifier()
    model.fit(train_features, train_label)
    return model

def train_decision_tree(train_features, train_label):
    model = DecisionTreeClassifier()
    model.fit(train_features, train_label)
    return model

def train_classifier_tree(train_features, train_label, **kwargs):
    estimators_key, estimators_default = 'n_estimators', 30
    depth_key, depth_default = 'max_depth', 40

    kwargs[estimators_key] = kwargs[estimators_key] if estimators_key in kwargs else estimators_default
    kwargs[depth_key] = kwargs[depth_key] if depth_key in kwargs else depth_default

    model = RandomForestClassifier(**kwargs)
    model.fit(train_features, train_label)
    return model
