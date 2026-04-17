"""
src/core/decision_engine.py
===========================
Green-Ops Framework — Decision Engine

CHANGES (v2):
  - FIX: decide() called gatekeeper.predict_failure_prob(similarity, change_size)
         but the original Gatekeeper signature was (similarity, change_size) —
         now extended to pass module_impact_score and other structural signals.
  - FIX: carbon_intensity was accepted as a parameter but never used in the
         decision logic — it was only passed to the LLM. Now properly forwarded
         to Gatekeeper.compare_to_carbon_threshold().
  - NEW: decide() now returns a richer dict including carbon threshold comparison
         result, so callers can see exactly why a test was pruned or kept.
  - IMPROVEMENT: Settings.FAILURE_THRESHOLD renamed to FAILURE_THRESHOLD for
         clarity; carbon threshold exposed as CARBON_THRESHOLD.
  - IMPROVEMENT: LLMAgent.decide() failure no longer crashes the engine —
         graceful fallback to the Pf-only decision.
"""

import logging

from src.ml.gatekeeper import Gatekeeper
from src.ai.llm_agent import LLMAgent
from src.config.settings import settings

log = logging.getLogger("greenops.decision_engine")


class DecisionEngine:
    """
    Orchestrates the final test-run/prune decision by combining:
      1. XGBoost Gatekeeper Pf prediction
      2. Carbon intensity vs threshold comparison
      3. LLM agent refinement (AI-optimised decision)
    """

    def __init__(self):
        self.gatekeeper = Gatekeeper(pf_threshold=settings.FAILURE_THRESHOLD)
        self.llm        = LLMAgent()

    def decide(
        self,
        similarity:           float,
        change_size:          int,
        carbon_intensity:     float,
        module_impact_score:  float = 0.5,
        is_kafka_consumer:    int   = 0,
        is_kafka_producer:    int   = 0,
        is_shared_db:         int   = 0,
        is_frontend_contract: int   = 0,
        is_shared_utility:    int   = 0,
        transitive_depth:     int   = 1,
        test_name:            str   = "",
    ) -> dict:
        """
        Produce a run/prune decision for a single (module, test) pair.

        Args:
            similarity           : cosine similarity between module and test embeddings
            change_size          : number of lines changed in the PR diff
            carbon_intensity     : live grid intensity in gCO2/kWh
            module_impact_score  : LLM impact confidence from Step 2 (0–1)
            is_kafka_*           : binary dependency flags
            is_shared_db/utility : binary dependency flags
            transitive_depth     : hops in dependency graph
            test_name            : optional label for logging/debugging

        Returns dict with keys:
            decision              : "RUN_ALL_TESTS" | "RUN_TEST" | "PRUNE_TEST"
            reason                : human-readable explanation
            probability           : Pf score (0–1)
            carbon_check          : full carbon threshold comparison dict
            ai_decision           : raw LLM decision string (or "unavailable")
        """
        # ── Step 1: Predict Pf ────────────────────────────────────────────────
        pf = self.gatekeeper.predict_failure_prob(
            similarity           = similarity,
            change_size          = change_size,
            module_impact_score  = module_impact_score,
            is_kafka_consumer    = is_kafka_consumer,
            is_kafka_producer    = is_kafka_producer,
            is_shared_db         = is_shared_db,
            is_frontend_contract = is_frontend_contract,
            is_shared_utility    = is_shared_utility,
            transitive_depth     = transitive_depth,
        )

        # ── Step 2: Carbon threshold comparison ───────────────────────────────
        carbon_check = self.gatekeeper.compare_to_carbon_threshold(
            pf               = pf,
            carbon_intensity = carbon_intensity,
            test_name        = test_name,
        )

        # ── Step 3: Safety rule — always run on very high Pf ─────────────────
        if pf >= settings.FAILURE_THRESHOLD:
            return {
                "decision":    "RUN_ALL_TESTS",
                "reason":      (
                    f"High failure probability (Pf={pf:.3f} ≥ "
                    f"threshold={settings.FAILURE_THRESHOLD:.3f})"
                ),
                "probability": round(pf, 4),
                "carbon_check": carbon_check,
                "ai_decision": "overridden by safety rule",
            }

        # ── Step 4: AI refinement (optional) ─────────────────────────────────
        ai_decision = "unavailable"
        try:
            ai_decision = self.llm.decide(similarity, carbon_intensity)
        except Exception as exc:
            log.warning("LLM agent unavailable: %s — using Pf-only decision", exc)

        # ── Step 5: Final decision ────────────────────────────────────────────
        final_run    = carbon_check["run"]
        final_reason = carbon_check["reason"]
        if not final_run:
            decision = "PRUNE_TEST"
        else:
            decision = "RUN_TEST"

        return {
            "decision":     decision,
            "reason":       final_reason,
            "probability":  round(pf, 4),
            "carbon_check": carbon_check,
            "ai_decision":  ai_decision,
        }
