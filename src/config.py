import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent
GREENOPS_DB   = os.environ.get("GREENOPS_DB", str(BASE_DIR / "greenops.db"))
OUTPUT_DIR    = Path(os.environ.get("GREENOPS_OUTPUT", str(BASE_DIR / "greenops_output")))
GREENOPS_OUTPUT = OUTPUT_DIR
MODEL_CACHE   = Path(os.environ.get("GREENOPS_MODEL_CACHE", str(BASE_DIR / "model_cache")))

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "5001"))

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY           = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY           = os.environ.get("OPENAI_API_KEY", "")
GITHUB_TOKEN             = os.environ.get("GITHUB_TOKEN", "")
ELECTRICITY_MAPS_API_KEY = os.environ.get("ELECTRICITY_MAPS_API_KEY", "")
ANTHROPIC_MODEL          = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

# Pull Request Target (for simulation)
GITHUB_REPO = os.environ.get("GITHUB_REPO", "akshaya209/AI-powered-carbon-aware-ci-cd-pipeline-optimiser")
GITHUB_PR   = int(os.environ.get("GITHUB_PR", "1"))

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE SPECIFICATIONS (Runner-specific)
# ─────────────────────────────────────────────────────────────────────────────

CPU_FREQUENCY_GHZ = float(os.environ.get("CPU_FREQUENCY_GHZ", "3.0"))
CPU_TDP_WATTS     = float(os.environ.get("CPU_TDP_WATTS", "15.0"))
JOULES_TO_KWH     = 1 / 3_600_000

# Datasets
PRESUBMIT_CSV        = os.environ.get("PRESUBMIT_CSV",  str(BASE_DIR / "presubmit_clean.csv"))
POSTSUBMIT_CSV       = os.environ.get("POSTSUBMIT_CSV", str(BASE_DIR / "postsubmit_clean.csv"))

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# Semantic Similarity (GraphCodeBERT)
SIMILARITY_THRESHOLD    = float(os.environ.get("SIMILARITY_THRESHOLD", "0.75"))
EMBEDDING_SIM_THRESHOLD = float(os.environ.get("EMBEDDING_SIM_THRESHOLD", "0.72"))
EMBEDDING_DIM           = int(os.environ.get("EMBEDDING_DIM", "768"))

# XGBoost Gatekeeper
PF_THRESHOLD  = float(os.environ.get("PF_THRESHOLD", "0.45"))
XGBOOST_PARAMS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
}

# Carbon Metrics
DIRTY_GRID_THRESHOLD = float(os.environ.get("DIRTY_GRID_THRESHOLD", "500"))
DEFER_CARBON_SCORE   = float(os.environ.get("DEFER_CARBON_SCORE", "0.65"))
DEFAULT_ZONE         = os.environ.get("DEFAULT_CARBON_ZONE", "IN-SO")
DEFAULT_CARBON_FALLBACK = 450  # Only used if API and DB both fail

# Scheduling Operation Thresholds
LIGHT_OPS_THRESHOLD  = 5000
MEDIUM_OPS_THRESHOLD = 20000
HEAVY_OPS_THRESHOLD  = 100000

# Simulation Baselines
BASELINE_CARBON_KG = 1.85

# Preprocessing Thresholds
MIN_PASS_RATE_PRE     = float(os.environ.get("MIN_PASS_RATE_PRE", "0.8"))
MAX_PASS_RATE_POST    = float(os.environ.get("MAX_PASS_RATE_POST", "0.5"))
FLAKINESS_WINDOW_SIZE = int(os.environ.get("FLAKINESS_WINDOW_SIZE", "100"))

# Heuristic coefficients for Pf calculation
COEFFICIENT_SIM = 0.5
COEFFICIENT_SIZE = 0.1

# Simulation Scaling Constants
KWH_PER_TEST = 0.015
REPO_TIERS = {
    "tensorflow": 100.0,
    "landscape": 25.0,
    "greenops": 5.0,
    "default": 1.0
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

ML_FEATURE_COLUMNS = [
    "test_duration", "duration_log", "duration_zscore", "duration_normalized",
    "test_duration_mean", "test_duration_std", "test_duration_ratio",
    "test_pass_rate", "test_failure_rate", "test_total_runs", "test_total_failures",
    "test_flakiness_score", "build_pass_rate", "build_test_count",
    "build_failure_count", "build_has_failure", "is_duration_outlier",
    "test_is_unit", "test_is_integration", "test_is_e2e", "test_is_perf",
    "test_is_smoke", "test_name_length", "test_name_depth", "build_depth",
    "cosine_similarity", "change_size", "module_impact_score",
    "is_kafka_consumer", "is_kafka_producer", "is_shared_db",
    "is_frontend_contract", "is_shared_utility", "transitive_depth",
]
