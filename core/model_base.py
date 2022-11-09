from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def fit(self, train_features, train_labels):
        pass

    @abstractmethod
    def predict(self, test_features):
        pass

    @abstractmethod
    def score(self, test_features, test_labels):
        pass
