import xgboost as xgb
import numpy as np

class Gatekeeper:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1
        )
        self._train()

    def _train(self):
        # Temporary dataset (replace later with real data)
        X = np.array([
            [0.9, 10],
            [0.2, 50],
            [0.7, 20],
            [0.1, 80],
            [0.5, 30]
        ])
        y = np.array([0, 1, 0, 1, 1])

        self.model.fit(X, y)

    def predict_failure_prob(self, similarity, change_size):
        return float(self.model.predict_proba([[similarity, change_size]])[0][1])