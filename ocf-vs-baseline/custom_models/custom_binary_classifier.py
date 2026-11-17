class CustomBinaryClassifier:
    def fit(self, x, y):
        raise NotImplementedError("fit method not implemented")

    def predict(self, x):
        raise NotImplementedError("predict method not implemented")

    def predict_proba(self, x):
        raise NotImplementedError("predict_proba method not implemented")
