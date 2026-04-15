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
        import pandas as pd
        import os
        
        # Dataset from: https://www.kaggle.com/datasets/akshyaaa/xgboost-pruning-dataset-based-on-historic-failure
        dataset_path = "presubmit_clean.csv"
        
        try:
            # Attempt to load the actual dataset if it has been downloaded
            df = pd.read_csv(dataset_path)
            
            # Use appropriate feature columns simulating 'similarity' and 'change_size'
            # If the CSV has specific ones, we can use them. Otherwise we fallback to what's available
            feature_cols = ['similarity', 'change_size'] if 'similarity' in df.columns else df.select_dtypes(include=[np.number]).columns[:2].tolist()
            target_col = 'test_result' if 'test_result' in df.columns else df.columns[-1]
            
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            self.model.fit(X, y)
            print(f"Gatekeeper successfully trained on {len(df)} records from {dataset_path}.")
        except Exception as e:
            print(f"Warning: Could not load {dataset_path} ({e}). Falling back to mock data.")
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