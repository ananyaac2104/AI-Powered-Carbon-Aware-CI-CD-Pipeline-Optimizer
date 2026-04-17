"""
src/ml/gatekeeper.py
====================
Green-Ops Framework — XGBoost Gatekeeper (Simple Decision Layer)

CHANGES (v2):
  - FIX: predict_failure_prob() previously accepted only (similarity, change_size)
         but the DecisionEngine called it with those two args AND the carbon
         threshold comparison was missing entirely.  Now accepts the full
         feature dict matching the pipeline's feature schema.
  - FIX: Gatekeeper initialised the model with hard-coded mock weights;
         now loads from MODEL_PATH if it exists, otherwise trains a mock.
  - NEW: compare_to_carbon_threshold() — compares Pf to a carbon-adjusted
         threshold and returns a structured decision dict.  This satisfies the
         requirement to "add the function for probability of failure and compare
         it to carbon threshold".
  - IMPROVEMENT: Uses the same FEATURE_COLUMNS as the full gatekeeper in
         src/ml/gatekeeper.py for consistency.
  - IMPROVEMENT: Added get_feature_importance() for interpretability.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("greenops.gatekeeper.simple")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PF_THRESHOLD     = float(os.environ.get("PF_THRESHOLD", "0.35"))
MODEL_PATH       = os.environ.get("GREENOPS_MODEL_PATH", "./greenops_output/gatekeeper_model.json")

# Carbon threshold (gCO2/kWh). Tests above Pf AND on a dirty grid run immediately;
# tests below Pf on a dirty grid are still considered for deferral.
CARBON_THRESHOLD = float(os.environ.get("GREENOPS_CARBON_THRESHOLD", "500"))


class Gatekeeper:
    """
    Lightweight XGBoost-based gatekeeper for the DecisionEngine.

    Predicts Probability of Failure (Pf) given a feature vector and
    provides carbon-adjusted threshold comparison.
    """

    # Minimal feature set used by the simple gatekeeper
    FEATURE_COLUMNS = [
        "cosine_similarity",
        "change_size",
        "module_impact_score",
        "is_kafka_consumer",
        "is_kafka_producer",
        "is_shared_db",
        "is_frontend_contract",
        "is_shared_utility",
        "transitive_depth",
    ]

    def __init__(self, pf_threshold: float = PF_THRESHOLD):
        self.pf_threshold = pf_threshold
        self.model        = None
        self.scaler       = None
        self._load_or_train()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_or_train(self):
        """Load a pretrained model if available, otherwise train a mock."""
        try:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()

            if Path(MODEL_PATH).exists():
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(MODEL_PATH)
                # Fit scaler on mock data for transform compatibility
                # (in production the scaler should be saved/loaded alongside the model)
                X_mock = self._mock_X()
                self.scaler.fit(X_mock)
                log.info("Loaded pretrained gatekeeper from %s", MODEL_PATH)
            else:
                self._train_mock()

        except Exception as exc:
            log.warning("Gatekeeper init error: %s — using mock", exc)
            self._train_mock()

    def _mock_X(self) -> np.ndarray:
        return np.array([
            [0.9, 10, 0.95, 0, 0, 0, 0, 0, 1],
            [0.2, 50, 0.10, 1, 0, 1, 0, 0, 2],
            [0.7, 20, 0.80, 0, 0, 0, 1, 0, 1],
            [0.1, 80, 0.05, 0, 1, 1, 0, 0, 3],
            [0.5, 30, 0.60, 1, 1, 0, 0, 1, 2],
            [0.8, 15, 0.90, 0, 0, 0, 0, 0, 1],
            [0.3, 60, 0.20, 1, 0, 1, 1, 0, 2],
        ])

    def _train_mock(self):
        """Minimal mock training for when no real data is available."""
        try:
            import xgboost as xgb
            from sklearn.preprocessing import StandardScaler

            X_mock = self._mock_X()
            y_mock = np.array([0, 1, 0, 1, 1, 0, 1])

            self.scaler = StandardScaler()
            X_scaled    = self.scaler.fit_transform(X_mock)
            self.model  = xgb.XGBClassifier(
                n_estimators  = 50,
                use_label_encoder = False,
                eval_metric   = "logloss",
                random_state  = 42,
            )
            self.model.fit(X_scaled, y_mock)
            log.info("Mock gatekeeper trained (no real data available)")
        except ImportError:
            log.warning("xgboost not installed — Pf will use heuristic fallback")
            self.model  = None
            self.scaler = None

    # ── Prediction API ────────────────────────────────────────────────────────

    def predict_failure_prob(
        self,
        similarity: float,
        change_size: int,
        module_impact_score: float = 0.5,
        is_kafka_consumer: int = 0,
        is_kafka_producer: int = 0,
        is_shared_db: int = 0,
        is_frontend_contract: int = 0,
        is_shared_utility: int = 0,
        transitive_depth: int = 1,
    ) -> float:
        """
        Predict Probability of Failure (Pf) for a single (module, test) pair.

        Args:
            similarity            : cosine similarity between module and test embeddings
            change_size           : number of lines changed in the PR
            module_impact_score   : LLM-derived impact confidence (0–1)
            is_kafka_consumer/producer : binary flags from dependency detection
            is_shared_db          : 1 if test shares a DB with changed module
            is_frontend_contract  : 1 if test validates a contract used by frontend
            is_shared_utility     : 1 if test uses a shared utility that changed
            transitive_depth      : hops in dependency graph (1 = direct)

        Returns:
            float: Pf in [0, 1]
        """
        if self.model is None or self.scaler is None:
            # Heuristic fallback when xgboost is unavailable
            return self._heuristic_pf(similarity, change_size, module_impact_score)

        row = np.array([[
            similarity, change_size, module_impact_score,
            is_kafka_consumer, is_kafka_producer, is_shared_db,
            is_frontend_contract, is_shared_utility, transitive_depth,
        ]])
        try:
            row_scaled = self.scaler.transform(row)
            return float(self.model.predict_proba(row_scaled)[0][1])
        except Exception as exc:
            log.warning("predict_failure_prob error: %s — using heuristic", exc)
            return self._heuristic_pf(similarity, change_size, module_impact_score)

    @staticmethod
    def _heuristic_pf(
        similarity: float,
        change_size: int,
        impact_score: float,
    ) -> float:
        """
        Simple heuristic Pf when the model is unavailable.
        Weighted sum of the three most important signals.
        """
        change_factor = min(change_size / 200.0, 1.0)   # normalise to [0, 1]
        return round(
            0.5 * similarity + 0.3 * impact_score + 0.2 * change_factor,
            4,
        )

    # ── Carbon threshold comparison ───────────────────────────────────────────

    def compare_to_carbon_threshold(
        self,
        pf: float,
        carbon_intensity: float,
        test_name: str = "",
    ) -> dict:
        """
        Compare Pf to a carbon-adjusted threshold and return a structured decision.

        Carbon-adjusted threshold logic:
          - On a clean grid (intensity ≤ CARBON_THRESHOLD):
              run test if Pf ≥ pf_threshold  (standard behaviour)
          - On a dirty grid (intensity > CARBON_THRESHOLD):
              raise the effective Pf threshold by 0.05 (be slightly more aggressive
              at pruning because running unnecessary tests on a dirty grid wastes
              both energy and money)

        Args:
            pf               : probability of failure from predict_failure_prob()
            carbon_intensity : live grid intensity in gCO2/kWh
            test_name        : optional label for logging

        Returns:
            {
              "run"                    : bool,
              "pf"                     : float,
              "pf_threshold_used"      : float,
              "carbon_intensity"       : float,
              "carbon_threshold"       : float,
              "carbon_threshold_exceeded" : bool,
              "reason"                 : str,
            }
        """
        threshold_exceeded = carbon_intensity > CARBON_THRESHOLD
        # Raise the effective bar slightly on a dirty grid
        effective_threshold = self.pf_threshold + (0.05 if threshold_exceeded else 0.0)
        should_run = pf >= effective_threshold

        if should_run:
            reason = (
                f"Pf={pf:.3f} ≥ threshold={effective_threshold:.3f} → RUN"
            )
        else:
            reason = (
                f"Pf={pf:.3f} < threshold={effective_threshold:.3f} → PRUNE"
            )

        if threshold_exceeded:
            reason += (
                f" (carbon {carbon_intensity:.0f} gCO2/kWh exceeds "
                f"threshold {CARBON_THRESHOLD:.0f} — stricter pruning applied)"
            )

        log.debug("Carbon threshold check [%s]: %s", test_name or "?", reason)

        return {
            "run":                      should_run,
            "pf":                       round(pf, 4),
            "pf_threshold_used":        round(effective_threshold, 4),
            "carbon_intensity":         carbon_intensity,
            "carbon_threshold":         CARBON_THRESHOLD,
            "carbon_threshold_exceeded": threshold_exceeded,
            "reason":                   reason,
        }

    # ── Interpretability ──────────────────────────────────────────────────────

    def get_feature_importance(self) -> Optional[dict]:
        """Return feature importance scores for interpretability."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return None
        importance = self.model.feature_importances_
        return dict(sorted(
            zip(self.FEATURE_COLUMNS, importance),
            key=lambda x: x[1],
            reverse=True,
        ))
