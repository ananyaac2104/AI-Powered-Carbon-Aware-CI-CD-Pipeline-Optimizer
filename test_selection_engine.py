"""
test_selection_engine.py
========================
Green-Ops CI/CD Framework — Intelligent Test Selection Engine

This is the decision layer that combines:
  1. Embedding similarity scores (from pr_diff_processor.py)
  2. Dependency graph analysis (from dependency_graph_engine.py)
  3. Hash delta signals (file actually changed vs unchanged)
  4. XGBoost Pf predictions (from src/ml/gatekeeper.py)
  5. Historical failure rate (from preprocessing.py outputs)

Replaces ALL demo/hardcoded logic ("high similarity → run", "medium → depends")
with real deterministic + ML-based selection.

OUTPUT FORMAT:
  {
    "changed_modules":    [...]         # files changed in PR
    "similarity_scores":  {file: score} # semantic similarity
    "impacted_modules":   [...]         # modules transitively impacted
    "final_tests":        [...]         # EXACT test files to run
    "pruned_tests":       [...]         # tests explicitly pruned
    "explanations":       [{test, decision, reason, pf_score, sim_score}]
    "summary": {
      "total_tests_discovered": int,
      "tests_selected":         int,
      "tests_pruned":           int,
      "pruning_rate":           float,
      "carbon_intensity":       float,
      "selection_strategy":     str,
    }
  }

USAGE:
    from test_selection_engine import TestSelectionEngine
    engine = TestSelectionEngine(repo="org/repo", repo_root="/path")
    result = engine.select_tests(
        diff_text="...",
        pr_number=42,
    )
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger("greenops.test_selection")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Minimum cosine similarity to consider a test "impacted" by a changed module
SIM_THRESHOLD       = float(os.environ.get("GREENOPS_SIM_THRESHOLD", "0.65"))
# Pf threshold for XGBoost gatekeeper (tests below this are pruned)
PF_THRESHOLD        = float(os.environ.get("PF_THRESHOLD", "0.30"))
# Carbon threshold (gCO2/kWh)
CARBON_THRESHOLD    = float(os.environ.get("GREENOPS_CARBON_THRESHOLD", "500"))
# Min similarity below which we ALWAYS run (hash changed + low sim = high risk)
HIGH_RISK_SIM_BOUND = float(os.environ.get("GREENOPS_HIGH_RISK_SIM", "0.40"))
# Max impact score below which tests are candidates for pruning
IMPACT_SCORE_PRUNE  = float(os.environ.get("GREENOPS_IMPACT_PRUNE", "0.15"))


# ─────────────────────────────────────────────────────────────────────────────
# TEST CANDIDATE RECORD
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidate:
    """
    Holds all signals for a single test file, used to make the run/prune decision.
    """

    def __init__(self, test_path: str):
        self.test_path          = test_path
        self.max_sim_score      = 0.0   # highest similarity to any changed module
        self.avg_sim_score      = 0.0   # avg similarity across changed modules
        self.in_dependency_path = False # True if reachable via import graph
        self.transitive_depth   = 999   # BFS depth from changed module
        self.hash_delta         = False # True if directly tested module hash changed
        self.pf_score           = 0.0   # XGBoost probability of failure
        self.historical_fail_rate = 0.0 # from preprocessing.py outputs
        self.is_always_run      = False # historic failure flag
        self.triggered_by       = []    # which changed modules triggered it
        self.impact_score       = 0.0   # composite impact score

    def compute_composite_score(self) -> float:
        """
        Compute final composite score for ranking.
        Higher = more likely to need running.
        """
        sim_signal   = self.max_sim_score
        dep_signal   = 1.0 if self.in_dependency_path else 0.0
        hash_signal  = 1.0 if self.hash_delta else 0.0
        depth_signal = max(0.0, 1.0 - self.transitive_depth / 6.0)
        pf_signal    = self.pf_score
        hist_signal  = self.historical_fail_rate

        self.impact_score = round(
            0.25 * sim_signal   +
            0.20 * dep_signal   +
            0.15 * hash_signal  +
            0.10 * depth_signal +
            0.20 * pf_signal    +
            0.10 * hist_signal,
            4,
        )
        return self.impact_score

    def should_run(
        self,
        sim_threshold:     float = SIM_THRESHOLD,
        pf_threshold:      float = PF_THRESHOLD,
        carbon_intensity:  float = 0.0,
        carbon_threshold:  float = CARBON_THRESHOLD,
    ) -> Tuple[bool, str]:
        """
        Deterministic run/prune decision.
        Returns (run: bool, reason: str).
        """
        # Always-run tests (historic failures)
        if self.is_always_run:
            return True, "ALWAYS_RUN: historic failure rate above 20%"

        # Hash changed AND test is in direct dependency path → always run
        if self.hash_delta and self.in_dependency_path and self.transitive_depth <= 1:
            return True, (
                f"DIRECT_DEPENDENCY: hash changed + direct import path "
                f"(sim={self.max_sim_score:.3f})"
            )

        # High semantic similarity → run
        if self.max_sim_score >= sim_threshold:
            reason = f"HIGH_SIMILARITY: sim={self.max_sim_score:.3f} ≥ {sim_threshold}"
            if carbon_intensity > carbon_threshold:
                reason += f" (carbon={carbon_intensity:.0f} exceeds threshold={carbon_threshold:.0f})"
            return True, reason

        # Transitive dependency AND similarity above half threshold → run
        if self.in_dependency_path and self.max_sim_score >= sim_threshold * 0.5:
            return True, (
                f"TRANSITIVE_DEP: depth={self.transitive_depth}, "
                f"sim={self.max_sim_score:.3f}"
            )

        # XGBoost Pf above threshold → run
        if self.pf_score >= pf_threshold:
            return True, f"HIGH_PF: pf={self.pf_score:.3f} ≥ {pf_threshold}"

        # Composite impact score above prune boundary → run
        if self.impact_score >= IMPACT_SCORE_PRUNE:
            return True, f"COMPOSITE_IMPACT: score={self.impact_score:.3f}"

        # Below all thresholds → prune
        return False, (
            f"PRUNED: sim={self.max_sim_score:.3f} < {sim_threshold}, "
            f"pf={self.pf_score:.3f} < {pf_threshold}, "
            f"dep={'yes' if self.in_dependency_path else 'no'}, "
            f"impact={self.impact_score:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_xgboost_features(
    candidate:       TestCandidate,
    change_size:     int,
    carbon_intensity: float,
) -> Dict:
    """
    Build the feature dict that the XGBoost gatekeeper expects.
    Maps TestCandidate signals to Gatekeeper.FEATURE_COLUMNS.
    """
    return {
        "cosine_similarity":    candidate.max_sim_score,
        "change_size":          change_size,
        "module_impact_score":  candidate.impact_score,
        "is_kafka_consumer":    0,
        "is_kafka_producer":    0,
        "is_shared_db":         1 if "db" in candidate.test_path.lower() or
                                     "database" in candidate.test_path.lower() else 0,
        "is_frontend_contract": 1 if "contract" in candidate.test_path.lower() or
                                     "api" in candidate.test_path.lower() else 0,
        "is_shared_utility":    1 if "util" in candidate.test_path.lower() or
                                     "helper" in candidate.test_path.lower() else 0,
        "transitive_depth":     candidate.transitive_depth if candidate.transitive_depth < 999 else 5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL FAILURE RATE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_historical_failure_rates(
    greenops_output: str = "./greenops_output",
) -> Dict[str, float]:
    """
    Load test-level failure rates from combined_submit.csv (output of preprocessing.py).
    Returns: {test_name_stem: failure_rate}
    """
    combined_path = Path(greenops_output) / "combined_submit.csv"
    if not combined_path.exists():
        log.info("No combined_submit.csv found — historical failure rates unavailable")
        return {}

    try:
        import pandas as pd
        df = pd.read_csv(combined_path, usecols=["test_name", "pass_rate_pre"])
        df = df.dropna(subset=["test_name", "pass_rate_pre"])
        df["failure_rate"] = 1.0 - df["pass_rate_pre"]
        rates = {}
        for _, row in df.iterrows():
            stem = Path(str(row["test_name"])).stem
            rates[stem] = float(row["failure_rate"])
        log.info("Loaded historical failure rates for %d tests", len(rates))
        return rates
    except Exception as e:
        log.warning("Failed to load historical failure rates: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST SELECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectionEngine:
    """
    Orchestrates all signals to select the exact set of tests to run for a PR.
    Replaces all demo-based PRUNE/RUN logic.
    """

    def __init__(
        self,
        repo:      str,
        repo_root: str = ".",
        db_path:   str = "./greenops_output/module_registry.sqlite",
        graph_path: Optional[str] = None,
        greenops_output: str = "./greenops_output",
    ):
        self.repo      = repo
        self.repo_root = str(Path(repo_root).resolve())
        self.db_path   = db_path
        self.greenops_output = greenops_output

        # Lazy imports to avoid circular deps at module load time
        self._store         = None
        self._diff_proc     = None
        self._dep_engine    = None
        self._gatekeeper    = None
        self._graph_path    = graph_path or str(Path(greenops_output) / "dependency_graph.json")
        self._historical_rates: Optional[Dict[str, float]] = None

    def _init_components(self):
        """Lazily initialise all sub-components."""
        if self._store is None:
            from module_embedding_store import get_store
            self._store = get_store(self.db_path)

        if self._diff_proc is None:
            from pr_diff_processor import PRDiffProcessor
            self._diff_proc = PRDiffProcessor(
                repo      = self.repo,
                repo_root = self.repo_root,
                db_path   = self.db_path,
            )

        if self._dep_engine is None:
            from dependency_graph_engine import DependencyGraphEngine
            self._dep_engine = DependencyGraphEngine(repo_root=self.repo_root)
            if Path(self._graph_path).exists():
                self._dep_engine.load(self._graph_path)
            else:
                log.info("No cached graph found — building dependency graph ...")
                self._dep_engine.build(
                    repo      = self.repo,
                    save_path = self._graph_path,
                )

        if self._gatekeeper is None:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
            try:
                from src.ml.gatekeeper import Gatekeeper
                self._gatekeeper = Gatekeeper()
                log.info("Loaded XGBoost gatekeeper")
            except Exception as e:
                log.warning("Cannot load gatekeeper: %s — using heuristic Pf", e)
                self._gatekeeper = None

        if self._historical_rates is None:
            self._historical_rates = load_historical_failure_rates(self.greenops_output)

    def select_tests(
        self,
        diff_text:         str,
        pr_number:         int = 0,
        carbon_intensity:  float = 500.0,
        force_run_all:     bool = False,
    ) -> Dict:
        """
        Full test selection pipeline.

        Args:
            diff_text:        unified diff from github_ci_integration.py
            pr_number:        PR number
            carbon_intensity: live grid intensity (gCO2/kWh)
            force_run_all:    if True, skip pruning (safety override)

        Returns structured output with exact test files + explanations.
        """
        self._init_components()

        log.info("=== Test Selection Engine (PR #%d) ===", pr_number)

        # Step 1: Process the PR diff
        diff_result = self._diff_proc.process_diff(
            diff_text  = diff_text,
            pr_number  = pr_number,
        )

        changed_modules  = diff_result["changed_modules"]
        similarity_scores = diff_result["similarity_scores"]
        impact_scores    = diff_result["impact_scores"]
        hash_deltas      = diff_result["hash_deltas"]
        total_lines      = diff_result["total_lines_changed"]

        if not changed_modules:
            log.info("No changed modules — selecting all tests (safety)")
            return self._all_tests_result(pr_number, carbon_intensity, "no_changes_detected")

        # Step 2: Dependency graph lookup
        dep_result = self._dep_engine.get_tests_for_changed_modules(
            changed_files      = changed_modules,
            include_transitive = True,
        )
        candidate_test_paths = set(dep_result["all_tests"])
        transitive_modules   = dep_result["transitive_modules"]

        # Also add tests found by embedding similarity
        for key, sim in diff_result["source_test_similarities"].items():
            # key format: "src_file→test_file"
            parts = key.split("→")
            if len(parts) == 2:
                candidate_test_paths.add(parts[1])

        log.info("Total test candidates: %d", len(candidate_test_paths))

        if not candidate_test_paths:
            log.info("No test candidates found — selecting all tests (safety)")
            return self._all_tests_result(pr_number, carbon_intensity, "no_candidates_found")

        # Step 3: Build TestCandidate objects for each candidate
        candidates: Dict[str, TestCandidate] = {}
        for test_path in candidate_test_paths:
            c = TestCandidate(test_path)

            # Similarity signals
            sim_values = []
            for src_file, sim in similarity_scores.items():
                pair_key = f"{src_file}→{test_path}"
                if pair_key in diff_result["source_test_similarities"]:
                    sim_values.append(diff_result["source_test_similarities"][pair_key])
                    c.triggered_by.append(src_file)

            # Also use per-module similarity as proxy if no direct pair found
            if not sim_values and similarity_scores:
                sim_values = list(similarity_scores.values())

            c.max_sim_score = max(sim_values) if sim_values else 0.0
            c.avg_sim_score = sum(sim_values) / len(sim_values) if sim_values else 0.0

            # Dependency signals
            module_test_map = dep_result.get("module_test_map", {})
            for src, tests in module_test_map.items():
                if test_path in tests:
                    c.in_dependency_path = True
                    c.transitive_depth   = 0
                    break

            if not c.in_dependency_path and test_path in dep_result["transitive_tests"]:
                c.in_dependency_path = True
                c.transitive_depth   = 2  # transitive

            # Hash delta signal (did the directly-tested module change?)
            for src_file in c.triggered_by:
                if hash_deltas.get(src_file, {}).get("changed", False):
                    c.hash_delta = True
                    break

            # Historical failure rate
            test_stem = Path(test_path).stem
            c.historical_fail_rate = self._historical_rates.get(test_stem, 0.0)
            c.is_always_run = c.historical_fail_rate >= 0.20

            # Compute composite score
            c.compute_composite_score()

            # XGBoost Pf
            if self._gatekeeper is not None:
                try:
                    features = build_xgboost_features(c, total_lines, carbon_intensity)
                    c.pf_score = self._gatekeeper.predict_failure_prob(
                        similarity           = features["cosine_similarity"],
                        change_size          = features["change_size"],
                        module_impact_score  = features["module_impact_score"],
                        is_kafka_consumer    = features["is_kafka_consumer"],
                        is_kafka_producer    = features["is_kafka_producer"],
                        is_shared_db         = features["is_shared_db"],
                        is_frontend_contract = features["is_frontend_contract"],
                        is_shared_utility    = features["is_shared_utility"],
                        transitive_depth     = features["transitive_depth"],
                    )
                except Exception as e:
                    log.debug("Gatekeeper predict failed for %s: %s", test_path, e)
                    c.pf_score = c.impact_score  # fallback

            candidates[test_path] = c

        # Step 4: Apply run/prune decision
        final_tests:  List[str] = []
        pruned_tests: List[str] = []
        explanations: List[Dict] = []

        # Adjust thresholds based on carbon
        effective_sim_threshold = SIM_THRESHOLD
        if carbon_intensity > CARBON_THRESHOLD:
            effective_sim_threshold = SIM_THRESHOLD + 0.05
            log.info("Carbon %.0f > threshold %.0f — raising sim threshold to %.2f",
                     carbon_intensity, CARBON_THRESHOLD, effective_sim_threshold)

        for test_path, c in sorted(
            candidates.items(),
            key=lambda x: x[1].impact_score,
            reverse=True,
        ):
            if force_run_all:
                run, reason = True, "FORCE_RUN_ALL"
            else:
                run, reason = c.should_run(
                    sim_threshold    = effective_sim_threshold,
                    pf_threshold     = PF_THRESHOLD,
                    carbon_intensity = carbon_intensity,
                    carbon_threshold = CARBON_THRESHOLD,
                )

            if run:
                final_tests.append(test_path)
            else:
                pruned_tests.append(test_path)

            explanations.append({
                "test":         test_path,
                "decision":     "RUN" if run else "PRUNE",
                "reason":       reason,
                "pf_score":     round(c.pf_score, 4),
                "sim_score":    round(c.max_sim_score, 4),
                "impact_score": round(c.impact_score, 4),
                "in_dep_graph": c.in_dependency_path,
                "hash_changed": c.hash_delta,
                "triggered_by": c.triggered_by[:3],
            })

        total_discovered = len(candidates)
        pruning_rate = round(
            len(pruned_tests) / max(total_discovered, 1), 4
        )

        result = {
            "changed_modules":   changed_modules,
            "similarity_scores": similarity_scores,
            "hash_deltas":       {
                fp: {"old": d["old_hash"], "new": d["new_hash"], "changed": d["changed"]}
                for fp, d in hash_deltas.items()
            },
            "impacted_modules":  list(set(transitive_modules + changed_modules)),
            "final_tests":       sorted(final_tests),
            "pruned_tests":      sorted(pruned_tests),
            "explanations":      explanations,
            "summary": {
                "total_tests_discovered": total_discovered,
                "tests_selected":         len(final_tests),
                "tests_pruned":           len(pruned_tests),
                "pruning_rate":           pruning_rate,
                "carbon_intensity":       carbon_intensity,
                "carbon_threshold_exceeded": carbon_intensity > CARBON_THRESHOLD,
                "total_lines_changed":    total_lines,
                "num_changed_modules":    len(changed_modules),
                "selection_strategy":     "embedding_similarity+dependency_graph+xgboost",
            },
        }

        # Persist final selection for audit
        if self._store:
            self._store.log_pr_run(
                repo           = self.repo,
                pr_number      = pr_number,
                changed_files  = changed_modules,
                selected_tests = final_tests,
                pruned_tests   = pruned_tests,
                pruning_rate   = pruning_rate,
            )

        self._print_selection_summary(result)
        return result

    def _all_tests_result(
        self, pr_number: int, carbon_intensity: float, strategy: str
    ) -> Dict:
        """Return a result that runs all known tests (safety fallback)."""
        all_tests = self._dep_engine.test_files if self._dep_engine else []
        return {
            "changed_modules":   [],
            "similarity_scores": {},
            "hash_deltas":       {},
            "impacted_modules":  [],
            "final_tests":       sorted(all_tests),
            "pruned_tests":      [],
            "explanations":      [
                {"test": t, "decision": "RUN",
                 "reason": f"Safety: {strategy}", "pf_score": 1.0,
                 "sim_score": 1.0, "impact_score": 1.0,
                 "in_dep_graph": True, "hash_changed": False, "triggered_by": []}
                for t in all_tests
            ],
            "summary": {
                "total_tests_discovered":  len(all_tests),
                "tests_selected":          len(all_tests),
                "tests_pruned":            0,
                "pruning_rate":            0.0,
                "carbon_intensity":        carbon_intensity,
                "carbon_threshold_exceeded": carbon_intensity > CARBON_THRESHOLD,
                "total_lines_changed":     0,
                "num_changed_modules":     0,
                "selection_strategy":      strategy,
            },
        }

    @staticmethod
    def _print_selection_summary(result: Dict):
        s = result["summary"]
        print(f"\n{'='*65}")
        print(f"Test Selection Complete")
        print(f"{'='*65}")
        print(f"  Strategy          : {s['selection_strategy']}")
        print(f"  Changed modules   : {s['num_changed_modules']}")
        print(f"  Lines changed     : {s['total_lines_changed']}")
        print(f"  Carbon intensity  : {s['carbon_intensity']:.0f} gCO2/kWh"
              f"  {'⚠ EXCEEDED' if s['carbon_threshold_exceeded'] else '✓ OK'}")
        print()
        print(f"  Tests discovered  : {s['total_tests_discovered']}")
        print(f"  Tests to RUN      : {s['tests_selected']}")
        print(f"  Tests PRUNED      : {s['tests_pruned']}")
        print(f"  Pruning rate      : {s['pruning_rate']:.1%}")
        print()

        run_exps   = [e for e in result["explanations"] if e["decision"] == "RUN"]
        prune_exps = [e for e in result["explanations"] if e["decision"] == "PRUNE"]

        if run_exps:
            print("  Selected tests (top 10):")
            for e in sorted(run_exps, key=lambda x: x["impact_score"], reverse=True)[:10]:
                print(f"    ✅ {Path(e['test']).name:<40} "
                      f"sim={e['sim_score']:.3f}  pf={e['pf_score']:.3f}")

        if prune_exps:
            print(f"\n  Pruned tests (top 5):")
            for e in prune_exps[:5]:
                print(f"    🚫 {Path(e['test']).name:<40} {e['reason'][:60]}")

        print(f"\n  Changed modules:")
        for m in result["changed_modules"]:
            sim = result["similarity_scores"].get(m, "?")
            hd  = result["hash_deltas"].get(m, {}).get("changed", False)
            sim_str = f"{sim:.4f}" if isinstance(sim, float) else str(sim)
            print(f"    → {m}  sim_to_stored={sim_str}  hash_changed={hd}")
        print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# RETRAINING HELPER FOR XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def retrain_xgboost_from_history(
    store_db_path:    str = "./greenops_output/module_registry.sqlite",
    repo:             str = "",
    output_model_path: str = "./greenops_output/gatekeeper_model.json",
) -> bool:
    """
    Retrain the XGBoost gatekeeper using REAL historical data from:
      - pr_run_log (run/pruned test decisions)
      - combined_submit.csv (pass/fail outcomes)

    This creates a real training loop: PR decisions → outcomes → updated model.

    Returns True if retrained successfully.
    """
    try:
        import pandas as pd
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report
        from module_embedding_store import SQLiteEmbeddingStore

        store = SQLiteEmbeddingStore(db_path=store_db_path)
        history = store.get_pr_history(repo, limit=1000)

        if len(history) < 10:
            log.warning("Insufficient history for retraining (%d runs)", len(history))
            return False

        # Load test outcomes from combined_submit
        outcomes_path = Path("./greenops_output/combined_submit.csv")
        if not outcomes_path.exists():
            log.warning("combined_submit.csv not found — cannot retrain")
            return False

        outcomes_df = pd.read_csv(outcomes_path)

        # Build training rows: each row = (test features, label: regressed or not)
        rows = []
        for run in history:
            for test in run.get("selected_tests", []):
                test_stem = Path(test).stem
                outcome_row = outcomes_df[outcomes_df["test_name"].str.contains(
                    test_stem, na=False, regex=False
                )]
                if outcome_row.empty:
                    continue
                regression = int(outcome_row.iloc[0].get("regression_detected", 0))
                pass_rate  = float(outcome_row.iloc[0].get("pass_rate_pre", 1.0))
                rows.append({
                    "cosine_similarity":    0.7,  # placeholder — would come from store
                    "change_size":          50,
                    "module_impact_score":  1.0 - pass_rate,
                    "is_kafka_consumer":    0,
                    "is_kafka_producer":    0,
                    "is_shared_db":         int("db" in test.lower()),
                    "is_frontend_contract": int("api" in test.lower()),
                    "is_shared_utility":    int("util" in test.lower()),
                    "transitive_depth":     1,
                    "label":                regression,
                })

        if len(rows) < 10:
            log.warning("Not enough labeled samples (%d) for retraining", len(rows))
            return False

        df_train = pd.DataFrame(rows)
        feature_cols = [c for c in df_train.columns if c != "label"]
        X = df_train[feature_cols].values
        y = df_train["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
        )

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / max(pos, 1)

        model = xgb.XGBClassifier(
            n_estimators      = 200,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = "logloss",
            use_label_encoder = False,
            random_state      = 42,
        )
        model.fit(
            X_train, y_train,
            eval_set     = [(X_test, y_test)],
            verbose      = False,
        )

        y_pred = model.predict(X_test)
        log.info("Retrained XGBoost:\n%s", classification_report(y_test, y_pred, zero_division=0))

        model.save_model(output_model_path)
        log.info("XGBoost model saved → %s", output_model_path)
        return True

    except Exception as e:
        log.error("XGBoost retraining failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Green-Ops Test Selection Engine")
    parser.add_argument("--repo",       required=True, help="org/repo")
    parser.add_argument("--repo-root",  default=".",  help="Repository root")
    parser.add_argument("--diff",       default=None, help="Path to .diff file")
    parser.add_argument("--pr-number",  type=int, default=0)
    parser.add_argument("--carbon",     type=float, default=500.0,
                        help="Live carbon intensity gCO2/kWh")
    parser.add_argument("--output",     default="./greenops_output")
    parser.add_argument("--retrain",    action="store_true",
                        help="Retrain XGBoost from historical data")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.retrain:
        success = retrain_xgboost_from_history(
            store_db_path     = str(out_dir / "module_registry.sqlite"),
            repo              = args.repo,
            output_model_path = str(out_dir / "gatekeeper_model.json"),
        )
        print("Retraining:", "SUCCESS" if success else "FAILED/INSUFFICIENT_DATA")
        sys.exit(0)

    diff_text = ""
    if args.diff:
        diff_text = Path(args.diff).read_text()
    else:
        # Try git diff HEAD~1
        import subprocess
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD~1", "HEAD"],
                capture_output=True, text=True, cwd=args.repo_root,
            )
            diff_text = result.stdout
        except Exception:
            pass

    engine = TestSelectionEngine(
        repo      = args.repo,
        repo_root = args.repo_root,
        db_path   = str(out_dir / "module_registry.sqlite"),
        greenops_output = str(out_dir),
    )

    result = engine.select_tests(
        diff_text        = diff_text,
        pr_number        = args.pr_number,
        carbon_intensity = args.carbon,
    )

    out_file = out_dir / f"test_selection_pr{args.pr_number}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull results saved → {out_file}")
