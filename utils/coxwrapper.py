from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator

class CoxWrapper(BaseEstimator):
    def __init__(self):
        self.model = CoxPHFitter()

    def fit(self, X, y):
        # y doit être un DataFrame avec colonnes "time" et "event"
        df = X.copy()
        df["time"] = y["time"]
        df["event"] = y["event"]
        self.model.fit(df, duration_col="time", event_col="event")
        return self

    def predict(self, X):
        return self.model.predict_partial_hazard(X)

# Exemple pipeline
cox_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("cox", CoxWrapper())
])
