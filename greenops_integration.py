"""
greenops_integration.py
=======================
Green-Ops CI/CD Framework — Single-Step CI Integration Runner

This is the script invoked by the GitHub Actions workflow step
"Run Green-Ops full pipeline". It wires all components together
and produces all artifacts needed by github_actions_runner.py.

Replaces the inline python -c "..." block in the original workflow YAML
with a clean, testable, importable module.

Components called (in order):
  1. repo_module_extractor  → embed all modules, store in SQLite
  2. github_ci_integration  → fetch diff, AST parse, hash
  3. pr_diff_processor      → re-embed changed files, compute similarities
  4. dependency_graph_engine → build/load import graph
  5. llm_impact_analyzer    → semantic impact (LLM or heuristic)
  6. xgboost_gatekeeper     → Pf prediction, prune/run decisions
  7. test_selection_engine  → exact test file selection
  8. carbon_inference_engine → live carbon intensity
  9. carbon_aware_scheduler → schedule tests by carbon/zone

Outputs written to $GREENOPS_OUTPUT/:
  - preprocessing_artifacts_prN.json  (from github_ci_integration)
  - module_registry.sqlite            (from extractor)
  - dependency_graph.json             (from dep engine)
  - impact_analysis.json              (from llm_impact_analyzer)
  - pruning_decision.json             (from gatekeeper)
  - test_schedule.json                (from scheduler)
  - test_selection_prN.json           (from selection engine)
  - pipeline_report_prN.json          (master report)

USAGE in CI:
    python greenops_integration.py

ENV VARS:
    GITHUB_TOKEN, REPO_NAME, PR_NUMBER, GREENOPS_OUTPUT,
    GREENOPS_PROVIDER, CO2SIGNAL_API_KEY, PF_THRESHOLD,
    GREENOPS_CARBON_THRESHOLD, GREENOPS_SIM_THRESHOLD
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("greenops.integration")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN    = os.environ.get("GITHUB_TOKEN", "")
REPO_NAME       = os.environ.get("REPO_NAME", "")
PR_NUMBER       = int(os.environ.get("PR_NUMBER", "0"))
REPO_ROOT       = os.environ.get("GITHUB_WORKSPACE", ".")
GREENOPS_OUTPUT = os.environ.get("GREENOPS_OUTPUT", "./greenops_output")
PROVIDER        = os.environ.get("GREENOPS_PROVIDER", "aws")
CARBON_STATE    = os.environ.get("GREENOPS_CARBON_STATE", "Maharashtra")


def run_full_pipeline(
    repo:            str   = REPO_NAME,
    pr_number:       int   = PR_NUMBER,
    repo_root:       str   = REPO_ROOT,
    greenops_output: str   = GREENOPS_OUTPUT,
    provider:        str   = PROVIDER,
    carbon_state:    str   = CARBON_STATE,
) -> dict:
    """
    Execute the full Green-Ops pipeline end-to-end.
    Returns the master report dict.
    """
    t_total = time.time()
    out     = Path(greenops_output)
    out.mkdir(parents=True, exist_ok=True)

    timings = {}

    log.info("=" * 65)
    log.info("GREEN-OPS INTEGRATION PIPELINE")
    log.info("  Repo     : %s", repo)
    log.info("  PR       : #%d", pr_number)
    log.info("  Root     : %s", repo_root)
    log.info("  Output   : %s", greenops_output)
    log.info("=" * 65)

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Full repo module extraction
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[1/8] Repo module extraction ...")
    extraction_report = {}
    try:
        from repo_module_extractor import RepoModuleExtractor
        extractor = RepoModuleExtractor(
            repo_root = repo_root,
            db_path   = str(out / "module_registry.sqlite"),
        )
        extraction_report = extractor.run_full_extraction(
            repo      = repo,
            pr_number = pr_number,
        )
    except Exception as e:
        log.warning("[1/8] Module extraction failed: %s", e)
    timings["extraction_ms"] = round((time.time() - t0) * 1000)

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Diff fetch + AST + hashes (github_ci_integration)
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[2/8] Fetching PR diff + AST parsing ...")
    diff_text         = ""
    preproc_artifacts = {"modules": [], "pr_number": pr_number, "repo": repo}
    try:
        from github_ci_integration import (
            fetch_pr_diff, extract_files_from_diff,
            load_module_from_path, run_ast_parser,
            run_module_hash_generator, write_artifacts,
        )
        import tempfile

        if GITHUB_TOKEN and repo and pr_number:
            diff_text = fetch_pr_diff(repo, pr_number, GITHUB_TOKEN)
            raw_diff_path = out / f"raw_diff_pr{pr_number}.diff"
            raw_diff_path.write_text(diff_text)
            log.info("Diff: %d chars", len(diff_text))
        else:
            # Try git locally
            import subprocess
            res = subprocess.run(
                ["git", "diff", "HEAD~1", "HEAD"],
                capture_output=True, text=True, cwd=repo_root, timeout=30
            )
            diff_text = res.stdout
            log.info("Using local git diff: %d chars", len(diff_text))

        if diff_text:
            from ast_parser import ASTParser
            ast_parser_obj = ASTParser(repo_root=repo_root)
            # Create a minimal shim module
            class ASTParserShim:
                def parse_file(self, fp): return ast_parser_obj.parse_file(fp).to_dict()
            ast_mod = ASTParserShim()

            from module_db import generate_hash, store_module
            class ModuleDBShim:
                def generate_hash(self, ar): return generate_hash(ar)
                def store_module(self, mi):  return store_module(mi)
            db_mod = ModuleDBShim()

            with tempfile.TemporaryDirectory() as tmp:
                changed_files = extract_files_from_diff(diff_text, Path(tmp))
                ast_results   = run_ast_parser(ast_mod, changed_files)
                module_records = run_module_hash_generator(db_mod, ast_results, pr_number, repo)
            preproc_artifacts = json.loads(json.dumps(
                write_artifacts(pr_number, repo, module_records, diff_text),
                default=str,
            )) if False else {   # write_artifacts returns paths, read the JSON
                "modules": module_records,
                "pr_number": pr_number,
                "repo": repo,
            }
            preproc_path = out / f"preprocessing_artifacts_pr{pr_number}.json"
            if preproc_path.exists():
                with open(preproc_path) as f:
                    preproc_artifacts = json.load(f)
    except Exception as e:
        log.warning("[2/8] CI integration failed: %s", e)
    timings["diff_ast_ms"] = round((time.time() - t0) * 1000)

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Dependency graph
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[3/8] Dependency graph ...")
    graph_path = str(out / "dependency_graph.json")
    try:
        from dependency_graph_engine import DependencyGraphEngine
        dep_engine = DependencyGraphEngine(repo_root=repo_root)
        if Path(graph_path).exists():
            dep_engine.load(graph_path)
        else:
            dep_engine.build(repo=repo, save_path=graph_path)
    except Exception as e:
        log.warning("[3/8] Dependency graph failed: %s", e)
        dep_engine = None
    timings["dep_graph_ms"] = round((time.time() - t0) * 1000)

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: LLM impact analysis
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[4/8] LLM impact analysis ...")
    impact = None
    try:
        from llm_impact_analyzer import LLMImpactAnalyzer
        analyzer = LLMImpactAnalyzer()
        dep_graph_data = None
        if dep_engine:
            dep_graph_data = {
                "test_map":    dep_engine.test_map,
                "module_graph": dep_engine.module_graph,
            }
        impact = analyzer.analyze(
            changed_modules = preproc_artifacts.get("modules", []),
            diff_text       = diff_text,
            dep_graph       = dep_graph_data,
            pr_number       = pr_number,
        )
        log.info("Impact: risk=%s, kafka=%s, db=%s",
                 impact.risk_level,
                 impact.kafka_topics_affected[:3],
                 impact.shared_db_tables_affected[:3])
    except Exception as e:
        log.warning("[4/8] Impact analysis failed: %s", e)
    timings["impact_ms"] = round((time.time() - t0) * 1000)

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: Carbon intensity
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[5/8] Carbon intensity ...")
    carbon_result = {"intensity": 500, "source": "default", "zone": carbon_state}
    try:
        from carbon_inference_engine import CarbonIntensityClient
        client        = CarbonIntensityClient(state=carbon_state)
        carbon_result = client.fetch_intensity_with_source()
        log.info("Carbon: %d gCO2/kWh (%s)", carbon_result["intensity"], carbon_result["source"])
    except Exception as e:
        log.warning("[5/8] Carbon fetch failed: %s — using default 500", e)
    timings["carbon_ms"] = round((time.time() - t0) * 1000)
    carbon_intensity = float(carbon_result["intensity"])

    # ──────────────────────────────────────────────────────────────────
    # STEP 6: Load embeddings from store for gatekeeper
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[6/8] Loading embeddings for gatekeeper ...")
    changed_embeddings: dict = {}
    test_embeddings:    dict = {}
    try:
        from module_embedding_store import SQLiteEmbeddingStore
        store        = SQLiteEmbeddingStore(str(out / "module_registry.sqlite"))
        changed_paths = [m.get("filepath", m.get("file_path", ""))
                         for m in preproc_artifacts.get("modules", [])]

        for fp in changed_paths:
            rec = store.get(repo, fp)
            if rec and rec.get("embedding") is not None:
                changed_embeddings[fp] = rec["embedding"]

        # Load test embeddings
        all_mods = store.list_all(repo)
        for mod in all_mods:
            fp = mod["file_path"]
            if "test" in fp.lower() or "spec" in fp.lower():
                rec = store.get(repo, fp)
                if rec and rec.get("embedding") is not None:
                    test_embeddings[fp] = rec["embedding"]

        log.info("Embeddings: %d changed, %d test",
                 len(changed_embeddings), len(test_embeddings))
    except Exception as e:
        log.warning("[6/8] Embedding load failed: %s", e)
    timings["embed_load_ms"] = round((time.time() - t0) * 1000)

    # ──────────────────────────────────────────────────────────────────
    # STEP 7: XGBoost gatekeeper
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[7/8] XGBoost gatekeeper ...")
    pruning_decision = {
        "run": [], "prune": [], "pf_scores": {},
        "pruning_rate": 0.0, "historic_failure_tests": [],
    }
    try:
        from xgboost_gatekeeper import run_gatekeeper_pipeline
        changed_modules = preproc_artifacts.get("modules", [])
        change_size     = sum(
            len(diff_text.splitlines())
            for _ in [diff_text]
        ) if diff_text else 100

        pruning_decision = run_gatekeeper_pipeline(
            changed_modules    = changed_modules,
            module_registry    = extraction_report.get("modules", changed_modules),
            changed_embeddings = changed_embeddings,
            test_embeddings    = test_embeddings,
            change_size        = change_size,
            carbon_intensity   = carbon_intensity,
            greenops_output    = greenops_output,
        )
        log.info("Gatekeeper: run=%d, prune=%d, pruning_rate=%.1f%%",
                 len(pruning_decision["run"]),
                 len(pruning_decision["prune"]),
                 pruning_decision["pruning_rate"] * 100)
    except Exception as e:
        log.warning("[7/8] Gatekeeper failed: %s", e)
        # Use test_selection_engine as fallback
        try:
            from test_selection_engine import TestSelectionEngine
            sel_engine = TestSelectionEngine(
                repo            = repo,
                repo_root       = repo_root,
                db_path         = str(out / "module_registry.sqlite"),
                graph_path      = graph_path,
                greenops_output = greenops_output,
            )
            sel_result = sel_engine.select_tests(
                diff_text        = diff_text,
                pr_number        = pr_number,
                carbon_intensity = carbon_intensity,
            )
            pruning_decision = {
                "run":    sel_result["final_tests"],
                "prune":  sel_result["pruned_tests"],
                "pf_scores": {
                    e["test"]: e["pf_score"]
                    for e in sel_result.get("explanations", [])
                },
                "pruning_rate":           sel_result["summary"]["pruning_rate"],
                "historic_failure_tests": [
                    e["test"] for e in sel_result.get("explanations", [])
                    if "ALWAYS_RUN" in e.get("reason", "")
                ],
            }
            log.info("Fallback selection: run=%d, prune=%d",
                     len(pruning_decision["run"]),
                     len(pruning_decision["prune"]))
        except Exception as e2:
            log.error("Fallback selection also failed: %s", e2)
    timings["gatekeeper_ms"] = round((time.time() - t0) * 1000)

    # ──────────────────────────────────────────────────────────────────
    # STEP 8: Carbon-aware scheduling
    # ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("[8/8] Carbon-aware scheduling ...")
    schedule = {"schedule_now": [], "schedule_deferred": [], "historic_failure_tests": []}
    try:
        from carbon_aware_scheduler import CarbonAwareScheduler
        scheduler = CarbonAwareScheduler(provider=provider)
        schedule  = scheduler.schedule(pruning_decision)
        log.info("Schedule: now=%d, deferred=%d",
                 len(schedule.get("schedule_now", [])),
                 len(schedule.get("schedule_deferred", [])))
        with open(out / "test_schedule.json", "w") as f:
            json.dump(schedule, f, indent=2, default=str)
    except Exception as e:
        log.warning("[8/8] Scheduler failed: %s", e)
        # Write a minimal schedule from pruning_decision
        schedule["schedule_now"] = [
            {"test_name": t, "pf_score": pruning_decision["pf_scores"].get(t, 0.5),
             "tier": "standard", "total_ops": 10000, "carbon_gco2": 0.0}
            for t in pruning_decision["run"]
        ]
        with open(out / "test_schedule.json", "w") as f:
            json.dump(schedule, f, indent=2, default=str)
    timings["schedule_ms"] = round((time.time() - t0) * 1000)

    # ──────────────────────────────────────────────────────────────────
    # MASTER REPORT
    # ──────────────────────────────────────────────────────────────────
    total_ms = round((time.time() - t_total) * 1000)
    report = {
        "pipeline_version":   "greenops-v3-production",
        "repo":               repo,
        "pr_number":          pr_number,
        "changed_modules":    [m.get("filepath", m.get("file_path", ""))
                               for m in preproc_artifacts.get("modules", [])],
        "final_tests":        pruning_decision.get("run", []),
        "pruned_tests":       pruning_decision.get("prune", []),
        "summary": {
            "tests_selected":    len(pruning_decision.get("run", [])),
            "tests_pruned":      len(pruning_decision.get("prune", [])),
            "pruning_rate":      pruning_decision.get("pruning_rate", 0.0),
            "carbon_intensity":  carbon_intensity,
            "carbon_source":     carbon_result.get("source", ""),
            "carbon_threshold_exceeded": carbon_intensity > float(
                os.environ.get("GREENOPS_CARBON_THRESHOLD", "500")
            ),
        },
        "impact": impact.to_dict() if impact else {},
        "schedule_summary": {
            "now":      len(schedule.get("schedule_now", [])),
            "deferred": len(schedule.get("schedule_deferred", [])),
            "historic": len(schedule.get("historic_failure_tests", [])),
            "recommendation": schedule.get("recommendation", ""),
        },
        "timings_ms": {**timings, "total_ms": total_ms},
    }

    report_path = out / f"pipeline_report_pr{pr_number}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    _print_master_summary(report)
    log.info("Pipeline complete in %.1fs → %s", total_ms / 1000, report_path)
    return report


def _print_master_summary(report: dict):
    s = report["summary"]
    print(f"\n{'='*65}")
    print("GREEN-OPS PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f"  Repo     : {report['repo']}  PR #{report['pr_number']}")
    print(f"  Changed  : {len(report['changed_modules'])} modules")
    print(f"  Carbon   : {s['carbon_intensity']:.0f} gCO2/kWh — {s['carbon_source']}")
    print(f"  {'⚠ EXCEEDED' if s['carbon_threshold_exceeded'] else '✓ below threshold'}")
    print()
    print(f"  ✅ Tests to RUN  : {s['tests_selected']}")
    print(f"  🚫 Tests PRUNED  : {s['tests_pruned']}")
    print(f"  Pruning rate     : {s['pruning_rate']:.1%}")
    t = report.get("timings_ms", {})
    print(f"  Total time       : {t.get('total_ms', 0) / 1000:.1f}s")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Green-Ops Full CI Integration")
    parser.add_argument("--repo",     default=REPO_NAME)
    parser.add_argument("--pr",       type=int, default=PR_NUMBER)
    parser.add_argument("--root",     default=REPO_ROOT)
    parser.add_argument("--output",   default=GREENOPS_OUTPUT)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--state",    default=CARBON_STATE)
    args = parser.parse_args()

    if not args.repo:
        print("Error: --repo required (or set REPO_NAME env var)")
        sys.exit(1)

    run_full_pipeline(
        repo            = args.repo,
        pr_number       = args.pr,
        repo_root       = args.root,
        greenops_output = args.output,
        provider        = args.provider,
        carbon_state    = args.state,
    )
