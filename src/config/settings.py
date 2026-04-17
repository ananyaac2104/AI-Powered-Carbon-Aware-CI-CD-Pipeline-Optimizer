"""
src/config/settings.py
======================
Green-Ops Framework — Centralised Configuration

CHANGES (v2):
  - FIX: Missing CARBON_THRESHOLD — referenced in decision_engine but only
         defined in gatekeeper.py as a module-level constant. Centralised here.
  - FIX: ANTHROPIC_API_KEY was missing — the LLM agent in src/ai/llm_agent.py
         uses Anthropic Claude as primary provider.
  - IMPROVEMENT: All thresholds documented with reasoning.
"""

import os


class Settings:
    # ── API Keys ──────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
    CO2SIGNAL_API_KEY   = os.getenv("CO2SIGNAL_API_KEY", "")

    # ── Pruning thresholds ────────────────────────────────────────────────────
    # Pf ≥ FAILURE_THRESHOLD → always run (safety override)
    FAILURE_THRESHOLD   = float(os.getenv("PF_THRESHOLD", "0.7"))

    # Carbon intensity (gCO2/kWh) above which heavy tests are deferred.
    # Based on India average (~659) vs Tamil Nadu clean target (~493).
    # 500 is a conservative middle ground.
    CARBON_THRESHOLD    = float(os.getenv("GREENOPS_CARBON_THRESHOLD", "500"))

    # ── Similarity ────────────────────────────────────────────────────────────
    # Cosine similarity threshold for GraphCodeBERT test relevance
    EMBEDDING_SIM_THRESHOLD = float(os.getenv("EMBEDDING_SIM_THRESHOLD", "0.72"))

    # ── Cloud provider ────────────────────────────────────────────────────────
    GREENOPS_PROVIDER   = os.getenv("GREENOPS_PROVIDER", "aws").lower()

    # ── Output directory ──────────────────────────────────────────────────────
    GREENOPS_OUTPUT     = os.getenv("GREENOPS_OUTPUT", "./greenops_output")


settings = Settings()
