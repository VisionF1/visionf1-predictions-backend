from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

class RandomForestPredictor:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = joblib.load(filepath)