"""
pipeline_runner.py  (production rewrite)
=========================================
Green-Ops CI/CD Framework — Reusable Pipeline Entry Point

This is the file called by action.yml. It reads ONLY from environment
variables so it works identically across any repository, any region,
any cloud provider — no repo-specific logic anywhere.

Key changes from previous version:
  - stage_get_carbon() is fully region-agnostic (reads GREENOPS_CARBON_ZONE)
  - stage_get_carbon() infers zone from RUNNER_REGION if zone not set
  - GenerativeDependencyMapper reads carbon_zone from env, not hardcoded state
  - All outputs written via set_actions_output() for Actions consumption
  - Structured JSON log written to pipeline.log for artifact upload
  - Confidence gate: if pruner confidence < GREENOPS_CONFIDENCE_MIN,
    auto-falls back to FULL_RUN (never silently prunes when uncertain)
  - PR_NUMBER=0 is valid for local dry-runs (skips GitHub API calls)

USAGE:
  # Inside GitHub Actions (via action.yml):
  python pipeline_runner.py --repo org/repo --pr 42 --base main

  # Local dry-run with a diff file:
  python pipeline_runner.py --repo org/repo --diff path/to/changes.diff

  # Demo mode:
  python pipeline_runner.py --demo
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("greenops.pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = Path(os.environ.get("GREENOPS_OUTPUT", ".greenops/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to SQLite module registry (embeddings + hashes)
MODULE_DB  = os.environ.get("GREENOPS_MODULE_DB", ".greenops/module_registry.sqlite")

# Confidence gate — if pruner confidence < this, fall back to full run
CONFIDENCE_MIN = float(os.environ.get("GREENOPS_CONFIDENCE_MIN", "0.60"))

# ─────────────────────────────────────────────────────────────────────────────
# REGION → CARBON ZONE MAPPING
# Maps GitHub Actions / AWS / Azure / GCP runner region identifiers to
# Electricity Maps zone codes. No India-specific hardcoding — works globally.
# Extend this map as needed; it is the ONLY place region logic lives.
# ─────────────────────────────────────────────────────────────────────────────

RUNNER_REGION_TO_ZONE = {
    # GitHub-hosted runner labels
    "ubuntu-latest":          "GB",     # GitHub's default runners are UK-based
    "ubuntu-22.04":           "GB",
    "ubuntu-20.04":           "GB",
    # AWS regions
    "us-east-1":              "US-NY-NYIS",
    "us-east-2":              "US-MIDW-MISO",
    "us-west-1":              "US-CAL-CISO",
    "us-west-2":              "US-NW-PACW",
    "eu-west-1":              "IE",
    "eu-west-2":              "GB",
    "eu-central-1":           "DE",
    "eu-north-1":             "SE",
    "ap-south-1":             "IN-SO",
    "ap-south-2":             "IN-SO",
    "ap-southeast-1":         "SG",
    "ap-southeast-2":         "AU-NSW",
    "ap-northeast-1":         "JP-TK",
    "ap-east-1":              "HK",
    "ca-central-1":           "CA-ON",
    "sa-east-1":              "BR-CS",
    # Azure regions
    "eastus":                 "US-NY-NYIS",
    "westus":                 "US-CAL-CISO",
    "westeurope":             "NL",
    "northeurope":            "IE",
    "centralindia":           "IN-SO",
    "southindia":             "IN-SO",
    "westindia":              "IN-SO",
    "uksouth":                "GB",
    "germanywestcentral":     "DE",
    "japaneast":              "JP-TK",
    "australiaeast":          "AU-NSW",
    # GCP regions
    "us-central1":            "US-MIDW-MISO",
    "us-east1":               "US-SE-SERC",
    "us-west1":               "US-NW-PACW",
    "europe-west1":           "BE",
    "europe-west2":           "GB",
    "europe-west3":           "DE",
    "asia-south1":            "IN-SO",
    "asia-south2":            "IN-NO",
    "asia-southeast1":        "SG",
    "asia-northeast1":        "JP-TK",
    "australia-southeast1":   "AU-NSW",
}

# Static fallback intensities (gCO2/kWh) per Electricity Maps zone.
# Used when the live API is unavailable. Source: Ember 2024 annual averages.
# This replaces ALL India-specific and region-specific hardcoded constants.
ZONE_STATIC_INTENSITY = {
    "GB":              207.0,
    "DE":              380.0,
    "FR":               60.0,
    "US-CAL-CISO":     210.0,
    "US-NY-NYIS":      170.0,
    "US-MIDW-MISO":    520.0,
    "US-NW-PACW":      120.0,
    "US-SE-SERC":      420.0,
    "IN-SO":           493.0,
    "IN-NO":           740.0,
    "IN-WE":           659.0,
    "IN-EA":           650.0,
    "SG":              410.0,
    "AU-NSW":          630.0,
    "JP-TK":           480.0,
    "SE":               13.0,
    "NO":               26.0,
    "IE":              350.0,
    "NL":              290.0,
    "BE":              150.0,
    "BR-CS":           100.0,
    "CA-ON":            40.0,
    "HK":              650.0,
    "default":         450.0,   # global average
}


def resolve_carbon_zone() -> str:
    """
    Resolve the Electricity Maps zone to use for carbon intensity.
    Priority:
      1. GREENOPS_CARBON_ZONE env var (explicit override)
      2. RUNNER_REGION env var → lookup in RUNNER_REGION_TO_ZONE
      3. "default" (uses global average static fallback)
    """
    explicit = os.environ.get("GREENOPS_CARBON_ZONE", "").strip()
    if explicit:
        return explicit

    runner_region = os.environ.get("RUNNER_REGION", "").strip().lower()
    if runner_region:
        zone = RUNNER_REGION_TO_ZONE.get(runner_region)
        if zone:
            log.info("Inferred carbon zone %s from RUNNER_REGION=%s", zone, runner_region)
            return zone

    log.info("No carbon zone configured — using global average static fallback")
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# ACTIONS OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_actions_output(key: str, value: str):
    """Write a key=value pair to $GITHUB_OUTPUT (Actions multi-line safe)."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        delimiter = "EOF_GREENOPS"
        with open(output_file, "a") as f:
            f.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")
    else:
        # Local run — just log it
        log.info("OUTPUT %s=%s", key, value[:120] + "..." if len(value) > 120 else value)


def emit_actions_outputs(result: dict, schedule: dict):
    """
    Emit all outputs consumed by action.yml and downstream steps.
    Called after the full pipeline completes.
    """
    selected  = result.get("final_tests",   [])
    pruned    = result.get("pruned_tests",  [])
    total     = len(selected) + len(pruned)
    prun_rate = round(len(pruned) / max(total, 1), 3)

    test_cmd  = os.environ.get("GREENOPS_TEST_COMMAND", "pytest")
    if selected:
        full_cmd = f"{test_cmd} {' '.join(selected)}"
    else:
        full_cmd = test_cmd   # no tests selected → run nothing (fallback handles this)

    summary = result.get("summary", {})

    set_actions_output("test_command",       full_cmd)
    set_actions_output("selected_tests",     " ".join(selected))
    set_actions_output("pruned_tests",       " ".join(pruned))
    set_actions_output("pruning_rate",       str(prun_rate))
    set_actions_output("carbon_intensity",   str(summary.get("carbon_intensity", 0)))
    set_actions_output("carbon_saved_grams", str(schedule.get("carbon_saved_grams", 0)))
    set_actions_output("time_saved_seconds", str(schedule.get("time_saved_seconds", 0)))
    set_actions_output("strategy",           summary.get("selection_strategy", "UNKNOWN"))
    set_actions_output("confidence",         str(summary.get("confidence", 0)))

    # GitHub Actions matrix for parallel test execution
    matrix = {
        "include": [{"test": t} for t in selected]
    } if selected else {"include": []}
    set_actions_output("matrix", json.dumps(matrix))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────────────────────────────────────

def stage_get_diff(
    repo:        str,
    pr_number:   int,
    base_branch: str,
    diff_file:   Optional[str] = None,
) -> str:
    """
    Fetch the PR diff. Three modes:
      A. From a local diff file (--diff flag, for dry-runs)
      B. From git (inside Actions, after fetch-depth: 0 checkout)
      C. From GitHub REST API (local dev with GITHUB_TOKEN)
    """
    if diff_file and Path(diff_file).exists():
        log.info("Loading diff from file: %s", diff_file)
        return Path(diff_file).read_text(encoding="utf-8")

    # Mode B: git diff (works in Actions after fetch-depth:0)
    base_sha = os.environ.get("BASE_SHA", "")
    if base_sha:
        try:
            result = subprocess.run(
                ["git", "diff", base_sha, "HEAD"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                log.info("Got diff from git (BASE_SHA=%s, %d chars)",
                         base_sha[:8], len(result.stdout))
                return result.stdout
        except Exception as exc:
            log.warning("git diff failed: %s", exc)

    # Mode B fallback: diff against remote base branch
    try:
        result = subprocess.run(
            ["git", "diff", f"origin/{base_branch}...HEAD"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            log.info("Got diff from git (origin/%s, %d chars)",
                     base_branch, len(result.stdout))
            return result.stdout
    except Exception as exc:
        log.warning("git diff (base branch) failed: %s", exc)

    # Mode C: GitHub REST API
    token = os.environ.get("GITHUB_TOKEN", "")
    if token and repo and pr_number > 0:
        import requests
        url     = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/vnd.github.v3.diff",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                log.info("Got diff from GitHub API (%d chars)", len(resp.text))
                return resp.text
        except Exception as exc:
            log.warning("GitHub API diff fetch failed: %s", exc)

    raise RuntimeError(
        "Could not fetch diff. Ensure one of:\n"
        "  A. Pass --diff <file>\n"
        "  B. Run inside GitHub Actions with fetch-depth: 0\n"
        "  C. Set GITHUB_TOKEN + REPO_NAME + PR_NUMBER"
    )


def stage_extract_modules(repo_root: str, repo: str, pr_number: int = 0) -> dict:
    """Stage 1: Full repo module extraction + embedding (with SQLite cache)."""
    print("\n" + "─" * 60)
    print("STAGE 1: Module Extraction + Embedding Cache")
    print("─" * 60)
    from repo_module_extractor import RepoModuleExtractor
    extractor = RepoModuleExtractor(
        repo_root = repo_root,
        db_path   = MODULE_DB,
    )
    return extractor.run_full_extraction(
        repo      = repo,
        pr_number = pr_number,
    )


def stage_build_dependency_graph(repo_root: str, repo: str) -> str:
    """Stage 2: Build or load dependency graph."""
    print("\n" + "─" * 60)
    print("STAGE 2: Dependency Graph")
    print("─" * 60)
    from dependency_graph_engine import DependencyGraphEngine
    graph_path = str(OUTPUT_DIR / "dependency_graph.json")
    engine = DependencyGraphEngine(repo_root=repo_root)
    if Path(graph_path).exists():
        log.info("Loading cached dependency graph")
        engine.load(graph_path)
    else:
        engine.build(repo=repo, save_path=graph_path)
    return graph_path


def stage_get_carbon() -> dict:
    """
    Stage 3: Fetch live carbon intensity.

    Fully region-agnostic — zone resolved from:
      GREENOPS_CARBON_ZONE > RUNNER_REGION > default
    No India-specific or region-specific constants here.
    """
    print("\n" + "─" * 60)
    print("STAGE 3: Carbon Intensity")
    print("─" * 60)

    zone = resolve_carbon_zone()
    log.info("Resolved carbon zone: %s", zone)

    # Try live API first
    api_key = (
        os.environ.get("ELECTRICITY_MAPS_KEY", "") or
        os.environ.get("CO2SIGNAL_API_KEY", "")
    )

    if api_key:
        try:
            import requests
            resp = requests.get(
                f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}",
                headers={"auth-token": api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                data      = resp.json()
                intensity = float(data.get("carbonIntensity", 0))
                if intensity > 0:
                    log.info("Live carbon: %.0f gCO2/kWh (zone=%s, source=ElectricityMaps)",
                             intensity, zone)
                    return {
                        "intensity": intensity,
                        "zone":      zone,
                        "source":    "ElectricityMaps live",
                    }
        except Exception as exc:
            log.warning("ElectricityMaps API failed: %s — using static fallback", exc)

    # Static fallback (Ember 2024 annual averages, zone-aware)
    intensity = ZONE_STATIC_INTENSITY.get(zone, ZONE_STATIC_INTENSITY["default"])
    log.info("Static carbon fallback: %.0f gCO2/kWh (zone=%s)", intensity, zone)
    return {
        "intensity": intensity,
        "zone":      zone,
        "source":    f"Ember 2024 static ({zone})",
    }


def stage_select_tests(
    repo:             str,
    repo_root:        str,
    diff_text:        str,
    pr_number:        int,
    carbon_intensity: float,
) -> dict:
    """Stage 4: Intelligent test selection (embedding + dep graph + XGBoost)."""
    print("\n" + "─" * 60)
    print("STAGE 4: Test Selection")
    print("─" * 60)
    from test_selection_engine import TestSelectionEngine
    engine = TestSelectionEngine(
        repo            = repo,
        repo_root       = repo_root,
        db_path         = MODULE_DB,
        graph_path      = str(OUTPUT_DIR / "dependency_graph.json"),
        greenops_output = str(OUTPUT_DIR),
    )
    return engine.select_tests(
        diff_text        = diff_text,
        pr_number        = pr_number,
        carbon_intensity = carbon_intensity,
    )


def stage_schedule(pruning_decision: dict, carbon: dict) -> dict:
    """Stage 5: Carbon-aware scheduling."""
    print("\n" + "─" * 60)
    print("STAGE 5: Carbon-Aware Scheduling")
    print("─" * 60)
    try:
        from carbon_aware_scheduler import CarbonAwareScheduler
        scheduler = CarbonAwareScheduler(
            carbon_zone      = carbon["zone"],
            carbon_intensity = carbon["intensity"],
        )
        return scheduler.schedule(pruning_decision)
    except Exception as exc:
        log.warning("Scheduler failed (%s) — returning pruning decision as-is", exc)
        return pruning_decision


def stage_run_tests(selection_result: dict) -> int:
    """Stage 6: Execute selected tests."""
    print("\n" + "─" * 60)
    print("STAGE 6: Test Execution")
    print("─" * 60)

    selected = selection_result.get("final_tests", [])
    strategy = selection_result.get("summary", {}).get("selection_strategy", "UNKNOWN")

    test_cmd = os.environ.get("GREENOPS_TEST_COMMAND", "pytest")

    if strategy == "FULL_RUN" or not selected:
        log.info("Strategy=%s → running full test suite", strategy)
        cmd = test_cmd
    else:
        log.info("Running %d selected tests (strategy=%s)", len(selected), strategy)
        cmd = f"{test_cmd} {' '.join(selected)}"

    log.info("Executing: %s", cmd)
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def stage_post_pr_comment(
    repo:      str,
    pr_number: int,
    result:    dict,
    carbon:    dict,
    schedule:  dict,
):
    """Stage 7: Post summary comment on PR (skipped for local runs)."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token or pr_number == 0:
        return

    print("\n" + "─" * 60)
    print("STAGE 7: PR Comment")
    print("─" * 60)

    try:
        from github_actions_runner import post_pr_comment, build_pr_comment
        comment_body = build_pr_comment(
            repo     = repo,
            schedule = schedule,
            result   = result,
            carbon   = carbon,
        )
        post_pr_comment(repo, pr_number, comment_body)
    except Exception as exc:
        log.warning("PR comment failed: %s", exc)


def generate_pipeline_log(
    stages:    dict,
    result:    dict,
    carbon:    dict,
    schedule:  dict,
    elapsed_s: float,
):
    """Write structured pipeline.log JSON for artifact upload."""
    log_data = {
        "pipeline_version": "2.0",
        "elapsed_seconds":  round(elapsed_s, 1),
        "carbon":           carbon,
        "stages":           stages,
        "summary":          result.get("summary", {}),
        "selected_tests":   result.get("final_tests", []),
        "pruned_tests":     result.get("pruned_tests", []),
        "schedule":         {
            k: v for k, v in schedule.items()
            if k not in ("tests", "deferred_tests")  # keep log compact
        },
    }
    log_path = OUTPUT_DIR / "pipeline.log"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    log.info("Pipeline log written → %s", log_path)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE GATE
# ─────────────────────────────────────────────────────────────────────────────

def apply_confidence_gate(result: dict) -> dict:
    """
    If the pruner's confidence is below CONFIDENCE_MIN, override the selection
    to FULL_RUN. This prevents silent under-testing on low-confidence PRs.
    """
    confidence = result.get("summary", {}).get("confidence", 1.0)
    if confidence < CONFIDENCE_MIN:
        log.warning(
            "Pruner confidence %.2f < threshold %.2f — overriding to FULL_RUN",
            confidence, CONFIDENCE_MIN
        )
        result["summary"]["selection_strategy"] = "FULL_RUN"
        result["summary"]["fallback_reason"]    = (
            f"confidence={confidence:.2f} below minimum={CONFIDENCE_MIN}"
        )
        result["final_tests"] = []   # empty → stage_run_tests runs full suite
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Green-Ops CI Pipeline Runner")
    parser.add_argument("--repo",   default=os.environ.get("REPO_NAME", ""))
    parser.add_argument("--pr",     type=int, default=int(os.environ.get("PR_NUMBER", "0")))
    parser.add_argument("--base",   default=os.environ.get("BASE_BRANCH", "main"))
    parser.add_argument("--diff",   default=None, help="Path to a local diff file")
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--demo",   action="store_true")
    args = parser.parse_args()

    if args.demo:
        from main import demo_decision_engine, demo_step2_pipeline
        demo_decision_engine()
        demo_step2_pipeline()
        return

    if not args.repo:
        parser.error("--repo is required (or set REPO_NAME env var)")

    t_start    = time.time()
    repo_root  = os.getcwd()
    stages     = {}

    log.info("=" * 60)
    log.info("GREEN-OPS PIPELINE  repo=%s  pr=#%d", args.repo, args.pr)
    log.info("=" * 60)

    # ── Stage 1: Diff ─────────────────────────────────────────────────────────
    t0 = time.time()
    diff_text = stage_get_diff(args.repo, args.pr, args.base, args.diff)
    stages["diff_fetch_ms"] = round((time.time() - t0) * 1000)

    # ── Stage 2: Module extraction ────────────────────────────────────────────
    t0 = time.time()
    stage_extract_modules(repo_root, args.repo, args.pr)
    stages["module_extraction_ms"] = round((time.time() - t0) * 1000)

    # ── Stage 3: Dependency graph ─────────────────────────────────────────────
    t0 = time.time()
    stage_build_dependency_graph(repo_root, args.repo)
    stages["dep_graph_ms"] = round((time.time() - t0) * 1000)

    # ── Stage 4: Carbon intensity (region-agnostic) ───────────────────────────
    t0 = time.time()
    carbon = stage_get_carbon()
    stages["carbon_fetch_ms"] = round((time.time() - t0) * 1000)

    # ── Stage 5: Test selection ───────────────────────────────────────────────
    t0 = time.time()
    result = stage_select_tests(
        repo             = args.repo,
        repo_root        = repo_root,
        diff_text        = diff_text,
        pr_number        = args.pr,
        carbon_intensity = carbon["intensity"],
    )
    stages["test_selection_ms"] = round((time.time() - t0) * 1000)

    # ── Confidence gate ───────────────────────────────────────────────────────
    result = apply_confidence_gate(result)

    # ── Stage 6: Scheduling ───────────────────────────────────────────────────
    t0 = time.time()
    schedule = stage_schedule(result, carbon)
    stages["scheduling_ms"] = round((time.time() - t0) * 1000)

    # ── Stage 7: Execute tests (skip in Actions — handled by matrix job) ──────
    exit_code = 0
    if os.environ.get("GITHUB_ACTIONS") != "true":
        exit_code = stage_run_tests(result)

    # ── Stage 8: PR comment ───────────────────────────────────────────────────
    stage_post_pr_comment(args.repo, args.pr, result, carbon, schedule)

    # ── Emit Actions outputs ──────────────────────────────────────────────────
    emit_actions_outputs(result, schedule)

    # ── Write structured log ──────────────────────────────────────────────────
    generate_pipeline_log(stages, result, carbon, schedule, time.time() - t_start)

    # ── Final summary ─────────────────────────────────────────────────────────
    summary = result.get("summary", {})
    print("\n" + "=" * 60)
    print("GREEN-OPS COMPLETE")
    print("=" * 60)
    print(f"  Strategy          : {summary.get('selection_strategy', 'UNKNOWN')}")
    print(f"  Tests selected    : {summary.get('tests_selected', 0)}")
    print(f"  Tests pruned      : {summary.get('tests_pruned', 0)}")
    print(f"  Pruning rate      : {summary.get('pruning_rate', 0):.1%}")
    print(f"  Carbon intensity  : {carbon['intensity']:.0f} gCO2/kWh ({carbon['zone']})")
    print(f"  Carbon source     : {carbon['source']}")
    print(f"  Carbon saved      : {schedule.get('carbon_saved_grams', 0):.1f} gCO2")
    print(f"  Time saved        : {schedule.get('time_saved_seconds', 0):.0f}s")
    print(f"  Confidence        : {summary.get('confidence', 0):.2f}")
    print(f"  Pipeline time     : {time.time() - t_start:.1f}s")
    print(f"  Artifacts         : {OUTPUT_DIR.resolve()}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
