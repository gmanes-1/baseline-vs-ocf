import numpy as np
from .custom_binary_classifier import CustomBinaryClassifier


class MajorityClassifier(CustomBinaryClassifier):
    def __init__(self):
        super().__init__()
        self.most_common_label = None

    def fit(self, x, y):
        """Fit the model by finding the most common label in the training data."""
        counts = np.bincount(np.asarray(y, dtype=int))
        self.most_common_label = int(np.argmax(counts))
        return self

    def predict(self, x):
        """Predict the class labels for the input data."""
        return np.full(len(x), self.most_common_label, dtype=int)

    def predict_proba(self, x):
        """Predict the class probabilities for the input data."""
        proba = np.zeros((len(x), 2))
        proba[:, self.most_common_label] = 1.0
        return proba
