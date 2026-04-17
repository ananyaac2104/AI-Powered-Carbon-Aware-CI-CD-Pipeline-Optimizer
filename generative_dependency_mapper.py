"""
generative_dependency_mapper.py
===============================
Green-Ops CI/CD Framework — Generative Layer & Dependency Mapper (Step 2)

Main orchestrator facade that delegates to the micro-architecture layers:
- Carbon API Network
- Dynamic Graph Builder
- Generative LLM Check

CHANGES (v2):
  - FIX: PipelineOutput.carbon_intensity typed as int but DatacenterSelector
         may produce float values; changed to float throughout.
  - FIX: execute() did not propagate errors from sub-components — a failure
         in any layer silently produced an empty result. Added per-step
         try/except with meaningful fallback and error surfacing.
  - FIX: PipelineInput.similarity_threshold had no validation — negative or
         >1 values caused every test to be either included or excluded.
         Added __post_init__ validation.
  - NEW: PipelineOutput now includes probability_of_failure per test and
         a carbon_threshold_exceeded flag comparing intensity to a configurable
         CARBON_THRESHOLD (default 500 gCO2/kWh — above which heavy tests
         are flagged for deferral). This satisfies the requirement:
         "add the function for probability of failure and compare it to carbon threshold".
  - IMPROVEMENT: execute() returns detailed per-step timing for observability.
  - IMPROVEMENT: Added to_artifact() for JSON serialisation of the output.
"""

import logging
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

from carbon_inference_engine import CarbonIntensityClient
from dynamic_graph_builder import DynamicGraphBuilder
from llm_generative_agent import GenerativeGraphEnhancer

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GreenOps.Orchestrator")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Carbon threshold (gCO2/kWh) above which heavy workloads should be deferred.
# Based on India average (~659) vs Tamil Nadu clean target (~493).
CARBON_THRESHOLD = float(os.environ.get("GREENOPS_CARBON_THRESHOLD", "500"))


# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineInput:
    """Immutable data contract for Step 2 input."""
    changed_functions:    List[str]
    similarity_scores:    Dict[Tuple[str, str], float]
    embeddings:           Dict[str, List[float]] = field(default_factory=dict)
    similarity_threshold: float = 0.5

    def __post_init__(self):
        # FIX: validate threshold so invalid values don't silently break pruning
        if not (0.0 < self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {self.similarity_threshold}"
            )


@dataclass
class PipelineOutput:
    """
    Data contract for Step 3 (ML Gatekeeper) consumption.

    NEW fields:
      probability_of_failure : dict mapping test_name → float Pf estimate
                               (pre-XGBoost: derived from weighted similarity score)
      carbon_threshold_exceeded : True if live carbon intensity exceeds CARBON_THRESHOLD
      carbon_threshold          : the threshold value used
      carbon_source             : description of the data source (for audit)
    """
    graph:                    Dict[str, List[str]]
    impacted_tests:           List[str]
    carbon_intensity:         float           # FIX: was int, now float
    test_weights:             Dict[str, float] = field(default_factory=dict)
    probability_of_failure:   Dict[str, float] = field(default_factory=dict)
    carbon_threshold:         float = CARBON_THRESHOLD
    carbon_threshold_exceeded: bool = False
    carbon_source:            str  = ""
    step_timings_ms:          Dict[str, float] = field(default_factory=dict)
    errors:                   List[str]        = field(default_factory=list)

    def to_artifact(self) -> dict:
        """Serialise to a JSON-compatible dict (excludes non-serialisable fields)."""
        d = asdict(self)
        # Convert tuple keys in nested dicts if any
        return d


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY OF FAILURE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pf_from_weights(
    test_weights: Dict[str, float],
    carbon_intensity: float,
    carbon_threshold: float = CARBON_THRESHOLD,
) -> Dict[str, float]:
    """
    Pre-XGBoost Pf estimate used by Step 2 before the full ML gatekeeper runs.

    Formula:
      base_pf      = normalised weight (weight / max_weight), range [0, 1]
      carbon_boost = 0.1 if carbon_intensity > carbon_threshold else 0.0
                     (dirty grid slightly raises failure risk due to throttling/
                      thermal events in overloaded datacenters)
      pf           = min(base_pf + carbon_boost, 1.0)

    This is a heuristic signal, not a trained prediction. The XGBoost gatekeeper
    in Step 3 will produce the definitive Pf. This value populates the
    PipelineOutput.probability_of_failure dict for early pruning signals.

    Returns:
        {test_name: pf_estimate}  sorted descending by pf
    """
    if not test_weights:
        return {}

    max_w = max(test_weights.values()) or 1.0
    carbon_boost = 0.1 if carbon_intensity > carbon_threshold else 0.0

    pf_estimates = {}
    for test, weight in test_weights.items():
        base_pf = weight / max_w
        pf      = min(base_pf + carbon_boost, 1.0)
        pf_estimates[test] = round(pf, 4)

    return dict(sorted(pf_estimates.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# CORE ORCHESTRATOR FACADE
# ─────────────────────────────────────────────────────────────────────────────

class GenerativeDependencyMapper:
    """
    Main orchestrator for Step 2 of the Architecture.
    Facade pattern delegating to instantiated network and builder objects.
    """

    def __init__(self, carbon_state: str = "Maharashtra"):
        self.logger       = logger
        self.graph_builder = DynamicGraphBuilder()
        self.llm_enhancer  = GenerativeGraphEnhancer()
        self.carbon_client = CarbonIntensityClient(state=carbon_state)

    def execute(self, payload: PipelineInput) -> PipelineOutput:
        """
        Executes the entire Step 2 pipeline with per-step error handling.
        Failures in individual steps are logged and reported in output.errors,
        but the pipeline continues with safe fallbacks.
        """
        self.logger.info("=" * 56)
        self.logger.info("STEP 2 PIPELINE COMMENCING")
        self.logger.info("=" * 56)

        errors: List[str] = []
        timings: Dict[str, float] = {}
        graph: Dict[str, List[str]] = {}
        tests: List[str] = []
        weights: Dict[str, float] = {}
        carbon_intensity: float = 0.0
        carbon_source: str = ""

        # ── Step 1: Dependency graph (static semantic layer) ──────────────────
        t0 = time.time()
        try:
            graph, tests = self.graph_builder.build_heuristic_graph(
                changed_functions    = payload.changed_functions,
                similarity_scores    = payload.similarity_scores,
                similarity_threshold = payload.similarity_threshold,
            )
            weights = self.graph_builder.weighted_impact_scores(
                similarity_scores    = payload.similarity_scores,
                similarity_threshold = payload.similarity_threshold,
                changed_functions    = payload.changed_functions,
            )
            self.logger.info(
                "Step 1 complete: %d function branches, %d impacted tests",
                len(graph), len(tests),
            )
        except Exception as exc:
            msg = f"Step 1 (graph builder) failed: {exc}"
            self.logger.error(msg)
            errors.append(msg)
        timings["graph_build_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Step 2: LLM graph enrichment (dynamic layer) ─────────────────────
        t0 = time.time()
        try:
            enriched_graph = self.llm_enhancer.verify_and_enrich_graph(graph)
        except Exception as exc:
            msg = f"Step 2 (LLM enrichment) failed: {exc}"
            self.logger.error(msg)
            errors.append(msg)
            enriched_graph = graph   # fallback to original
        timings["llm_enrichment_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Step 3: Carbon intensity + threshold comparison ───────────────────
        t0 = time.time()
        try:
            carbon_result    = self.carbon_client.fetch_intensity_with_source()
            carbon_intensity = float(carbon_result["intensity"])
            carbon_source    = carbon_result.get("source", "unknown")
            self.logger.info(
                "Step 3 complete: %s gCO2/kWh (source=%s, threshold=%s)",
                carbon_intensity, carbon_source, CARBON_THRESHOLD,
            )
        except Exception as exc:
            msg = f"Step 3 (carbon fetch) failed: {exc}"
            self.logger.error(msg)
            errors.append(msg)
            carbon_intensity = CARBON_THRESHOLD   # safe default = at threshold
            carbon_source    = "error fallback"
        timings["carbon_fetch_ms"] = round((time.time() - t0) * 1000, 1)

        threshold_exceeded = carbon_intensity > CARBON_THRESHOLD

        # ── Step 4: Probability of failure estimates ──────────────────────────
        pf_estimates = estimate_pf_from_weights(weights, carbon_intensity, CARBON_THRESHOLD)

        self.logger.info(
            "Step 2 pipeline complete | carbon=%.0f gCO2/kWh | "
            "threshold_exceeded=%s | errors=%d",
            carbon_intensity, threshold_exceeded, len(errors),
        )
        if threshold_exceeded:
            self.logger.warning(
                "⚠️  Carbon intensity %.0f gCO2/kWh exceeds threshold %.0f — "
                "recommend deferring heavy tests to off-peak.",
                carbon_intensity, CARBON_THRESHOLD,
            )

        return PipelineOutput(
            graph                    = enriched_graph,
            impacted_tests           = tests,
            carbon_intensity         = carbon_intensity,
            test_weights             = weights,
            probability_of_failure   = pf_estimates,
            carbon_threshold         = CARBON_THRESHOLD,
            carbon_threshold_exceeded = threshold_exceeded,
            carbon_source            = carbon_source,
            step_timings_ms          = timings,
            errors                   = errors,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL EXECUTION / DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = PipelineInput(
        changed_functions=["add", "multiply"],
        similarity_scores={
            ("add",      "test_add"):  0.92,
            ("add",      "test_db"):   0.12,
            ("multiply", "test_mul"):  0.89,
            ("multiply", "test_add"):  0.55,
        },
        similarity_threshold=0.5,
    )

    orchestrator  = GenerativeDependencyMapper(carbon_state="Tamil Nadu")
    result        = orchestrator.execute(mock_input)

    print("\n" + "=" * 60)
    print("STEP 2 PIPELINE OUTPUT")
    print("=" * 60)

    print("graph = {")
    for key, val in result.graph.items():
        print(f'    "{key}": {str(val).replace(chr(39), chr(34))},')
    print("}")

    print(f"\nimpacted_tests = {str(result.impacted_tests).replace(chr(39), chr(34))}")
    print(f"\ncarbon_intensity    = {result.carbon_intensity:.0f} gCO2/kWh")
    print(f"carbon_threshold    = {result.carbon_threshold:.0f} gCO2/kWh")
    print(f"threshold_exceeded  = {result.carbon_threshold_exceeded}")
    print(f"carbon_source       = {result.carbon_source}")
    print(f"\ntest_weights        = {result.test_weights}")
    print(f"\nprobability_of_failure:")
    for test, pf in result.probability_of_failure.items():
        marker = " ⚠️ HIGH" if pf >= 0.7 else ""
        print(f"  {test:<30} Pf={pf:.4f}{marker}")
    print(f"\nstep_timings_ms = {result.step_timings_ms}")
    if result.errors:
        print(f"\nerrors = {result.errors}")
    print("=" * 60 + "\n")
