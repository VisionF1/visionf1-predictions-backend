from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class GradientBoostingPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)