"""
github_actions_runner.py
=========================
Green-Ops CI/CD Framework — GitHub Actions + CLI Test Runner

Final stage of the pipeline. Takes the carbon-aware schedule
(from carbon_aware_scheduler.py) and:

  1. Posts a PR comment summarising the Green-Ops decision
     (which tests run, which are pruned, which DC, carbon saved)
  2. Dynamically writes a GitHub Actions matrix job file so only
     the scheduled tests are executed in the CI workflow
  3. Triggers the tests via GitHub CLI (gh workflow run) for
     immediate local execution
  4. Updates PR labels  (greenops:pruned / greenops:deferred / greenops:clean)
  5. Posts a final status check back to the PR commit

This is the "output layer" that closes the loop started by
github_ci_integration.py.

Flow in the full pipeline:
  github_ci_integration.py       ← fetches diff + AST + hashes
    → graphcodebert_embeddings.py ← embeds modules
      → llm_impact_analyzer.py   ← detects dependencies
        → xgboost_gatekeeper.py  ← predicts Pf, prunes tests
          → carbon_aware_scheduler.py ← picks cleanest DC
            → github_actions_runner.py  ← THIS FILE: triggers CI

Dependencies:
    pip install requests PyGithub

Environment variables (same as github_ci_integration.py):
    GITHUB_TOKEN   — personal access token or Actions ${{ secrets.GITHUB_TOKEN }}
    REPO_NAME      — e.g. "your-org/your-repo"
    PR_NUMBER      — pull request number
    GREENOPS_OUTPUT — directory where schedule JSON was written
    GREENOPS_PROVIDER — cloud provider (aws/azure/gcp), default aws

Usage (local):
    python github_actions_runner.py

Usage (inside Actions — called by greenops_trigger.yml):
    Automatically invoked after xgboost_gatekeeper step.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("greenops.runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN    = os.environ.get("GITHUB_TOKEN", "")
REPO_NAME       = os.environ.get("REPO_NAME", "")
PR_NUMBER       = int(os.environ.get("PR_NUMBER", "0"))
OUTPUT_DIR      = Path(os.environ.get("GREENOPS_OUTPUT", "./greenops_output"))
PROVIDER        = os.environ.get("GREENOPS_PROVIDER", "aws").lower()

# Paths to schedule artifacts written by earlier pipeline stages
SCHEDULE_PATH   = OUTPUT_DIR / "test_schedule.json"
DECISION_PATH   = OUTPUT_DIR / "pruning_decision.json"
IMPACT_PATH     = OUTPUT_DIR / "impact_analysis.json"

GITHUB_API_BASE = "https://api.github.com"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — GITHUB API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
    }


def post_pr_comment(repo: str, pr_number: int, body: str) -> bool:
    """Post a comment on the pull request."""
    url  = f"{GITHUB_API_BASE}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.post(url, headers=_headers(), json={"body": body}, timeout=30)
    if resp.status_code == 201:
        log.info("PR comment posted (PR #%d)", pr_number)
        return True
    log.warning("Failed to post PR comment: %d %s", resp.status_code, resp.text[:200])
    return False


def update_pr_labels(repo: str, pr_number: int, labels: list) -> bool:
    """Add labels to the pull request."""
    url  = f"{GITHUB_API_BASE}/repos/{repo}/issues/{pr_number}/labels"
    resp = requests.post(url, headers=_headers(), json={"labels": labels}, timeout=30)
    if resp.status_code in (200, 201):
        log.info("Labels added to PR #%d: %s", pr_number, labels)
        return True
    log.warning("Failed to add labels: %d", resp.status_code)
    return False


def create_pr_labels_if_missing(repo: str):
    """Create the Green-Ops labels in the repo if they don't exist yet."""
    label_defs = [
        {"name": "greenops:pruned",   "color": "00b894",
         "description": "Tests pruned by Green-Ops XGBoost gatekeeper"},
        {"name": "greenops:deferred", "color": "fdcb6e",
         "description": "Heavy tests deferred to off-peak by carbon scheduler"},
        {"name": "greenops:clean",    "color": "74b9ff",
         "description": "All tests passed on clean energy grid"},
        {"name": "greenops:historic", "color": "e17055",
         "description": "Historic failure tests always run"},
    ]
    url = f"{GITHUB_API_BASE}/repos/{repo}/labels"
    for label in label_defs:
        resp = requests.post(url, headers=_headers(), json=label, timeout=15)
        if resp.status_code == 422:
            pass   # already exists
        elif resp.status_code == 201:
            log.info("Created label: %s", label["name"])


def post_commit_status(
    repo: str, sha: str, state: str, description: str, context: str = "Green-Ops/carbon"
) -> bool:
    """
    Post a commit status check.
    state: "pending" | "success" | "failure" | "error"
    """
    url  = f"{GITHUB_API_BASE}/repos/{repo}/statuses/{sha}"
    body = {
        "state":       state,
        "description": description[:140],   # GitHub limit
        "context":     context,
        "target_url":  f"https://github.com/{repo}/pull/{PR_NUMBER}",
    }
    resp = requests.post(url, headers=_headers(), json=body, timeout=30)
    if resp.status_code == 201:
        log.info("Commit status posted: %s — %s", state, description)
        return True
    log.warning("Failed to post commit status: %d", resp.status_code)
    return False


def get_pr_head_sha(repo: str, pr_number: int) -> Optional[str]:
    """Get the HEAD SHA of a PR."""
    url  = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(url, headers=_headers(), timeout=15)
    if resp.status_code == 200:
        return resp.json().get("head", {}).get("sha", "")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PR COMMENT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_pr_comment(
    schedule:         dict,
    pruning_decision: dict,
    impact:           Optional[dict] = None,
) -> str:
    """
    Build a rich Markdown PR comment summarising the full Green-Ops decision.
    """
    run_now    = schedule.get("schedule_now", [])
    deferred   = schedule.get("schedule_deferred", [])
    historic   = schedule.get("historic_failure_tests", [])
    pruned     = pruning_decision.get("prune", [])
    pf_scores  = pruning_decision.get("pf_scores", {})
    provider   = schedule.get("provider", PROVIDER).upper()
    zone       = schedule.get("selected_zone", "unknown")
    city       = schedule.get("selected_city", "")
    state      = schedule.get("selected_state", "")
    intensity  = schedule.get("carbon_intensity", 0)
    score      = schedule.get("carbon_score", 0)
    total_ops  = schedule.get("total_ops_estimated", 0)
    total_co2  = schedule.get("total_carbon_gco2", 0)
    pruning_rt = pruning_decision.get("pruning_rate", 0)

    # Carbon savings estimate vs running everything
    all_tests = len(run_now) + len(deferred) + len(historic) + len(pruned)
    carbon_savings_pct = round(pruning_rt * 100, 1)

    # Impact summary from LLM
    llm_summary  = ""
    kafka_topics = []
    db_tables    = []
    if impact:
        llm_summary  = impact.get("summary", "")
        kafka_topics = impact.get("kafka_topics_affected", [])
        db_tables    = impact.get("shared_db_tables_affected", [])

    lines = [
        "## 🌱 Green-Ops Carbon-Aware CI Report",
        "",
        f"**Datacenter:** `{provider}/{zone}` — {city}, {state}  ",
        f"**Grid intensity:** `{intensity:.0f} gCO₂/kWh` (carbon score: `{score:.3f}`)  ",
        f"**Estimated carbon:** `{total_co2:.6f} gCO₂` for `{total_ops:,}` operations",
        "",
        "---",
        "",
        "### 📊 Test Selection Summary",
        "",
        f"| Category | Count | % of total |",
        f"|---|---|---|",
        f"| ✅ Run now (Pf above threshold) | {len(run_now)} | "
        f"{len(run_now)/max(all_tests,1)*100:.0f}% |",
        f"| ⚡ Always run (historic failures) | {len(historic)} | "
        f"{len(historic)/max(all_tests,1)*100:.0f}% |",
        f"| ⏸ Deferred (heavy + dirty grid) | {len(deferred)} | "
        f"{len(deferred)/max(all_tests,1)*100:.0f}% |",
        f"| 🚫 Pruned by XGBoost (low Pf) | {len(pruned)} | "
        f"{carbon_savings_pct}% |",
        f"| **Total** | **{all_tests}** | **100%** |",
        "",
    ]

    if llm_summary:
        lines += [
            "### 🤖 Impact Analysis",
            "",
            f"> {llm_summary}",
            "",
        ]
        if kafka_topics:
            lines.append(f"**Kafka topics affected:** {', '.join(f'`{t}`' for t in kafka_topics)}")
        if db_tables:
            lines.append(f"**DB tables affected:** {', '.join(f'`{t}`' for t in db_tables)}")
        if kafka_topics or db_tables:
            lines.append("")

    # Tests running now
    if run_now:
        lines += [
            "### ✅ Tests Running Now",
            "",
            "| Test | Pf Score | Tier | Ops | Carbon (gCO₂) |",
            "|---|---|---|---|---|",
        ]
        for t in sorted(run_now, key=lambda x: x["pf_score"], reverse=True):
            lines.append(
                f"| `{t['test_name']}` | `{t['pf_score']:.3f}` | {t['tier']} "
                f"| {t['total_ops']:,} | `{t['carbon_gco2']:.8f}` |"
            )
        lines.append("")

    # Historic failure tests
    if historic:
        lines += [
            "### ⚡ Always-Run Tests (Historic Failures)",
            "",
            "> These tests run on every PR regardless of code changes.",
            "",
            "| Test | Failure Rate | Tier |",
            "|---|---|---|",
        ]
        for t in historic:
            rate = t.get("pf_score", t.get("failure_rate", "?"))
            rate_str = f"{rate:.0%}" if isinstance(rate, float) else str(rate)
            lines.append(f"| `{t['test_name']}` | {rate_str} | {t.get('tier', '?')} |")
        lines.append("")

    # Deferred tests
    if deferred:
        lines += [
            "### ⏸ Deferred Tests (Off-Peak)",
            "",
            "| Test | Reason |",
            "|---|---|",
        ]
        for t in deferred:
            reason = t.get("defer_reason", "Heavy test on dirty grid")[:80]
            lines.append(f"| `{t['test_name']}` | {reason} |")
        lines.append("")

    # Pruned tests
    if pruned:
        lines += [
            "<details>",
            f"<summary>🚫 {len(pruned)} Pruned Tests (click to expand)</summary>",
            "",
            "| Test | Pf Score |",
            "|---|---|",
        ]
        for t in pruned[:20]:
            pf = pf_scores.get(t, "?")
            pf_str = f"`{pf:.3f}`" if isinstance(pf, float) else str(pf)
            lines.append(f"| `{t}` | {pf_str} |")
        if len(pruned) > 20:
            lines.append(f"| ... and {len(pruned)-20} more | — |")
        lines += ["", "</details>", ""]

    # Recommendation
    recommendation = schedule.get("recommendation", "")
    if recommendation:
        lines += [
            "---",
            "",
            f"**💡 Recommendation:** {recommendation}",
            "",
        ]

    # Footer
    lines += [
        "---",
        "",
        "*Powered by [Green-Ops](https://github.com/your-org/greenops) — "
        "Carbon-Aware CI/CD Framework*  ",
        f"*Data: Ember Global Electricity Review 2024 (CC BY 4.0)*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GITHUB ACTIONS MATRIX WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_test_matrix(schedule: dict, output_path: Path):
    """
    Write a JSON matrix file that GitHub Actions uses to run only
    the scheduled tests. Referenced in the workflow YAML as:
        matrix: ${{ fromJson(steps.greenops.outputs.matrix) }}

    Also writes a flat list of test names for bash-based test runners.
    """
    run_now  = [e["test_name"] for e in schedule.get("schedule_now", [])]
    historic = [e["test_name"] for e in schedule.get("historic_failure_tests", [])]

    # Combine: run now + historic (always run)
    # Deferred tests are written to a separate file for the off-peak job
    all_immediate = list(dict.fromkeys(run_now + historic))   # deduplicate, preserve order
    deferred_list = [e["test_name"] for e in schedule.get("schedule_deferred", [])]

    # GitHub Actions matrix format
    matrix = {
        "include": [
            {
                "test":     test_name,
                "provider": schedule.get("provider", PROVIDER),
                "zone":     schedule.get("selected_zone", "ap-south-1"),
                "city":     schedule.get("selected_city", "Mumbai"),
            }
            for test_name in all_immediate
        ]
    }

    matrix_path    = output_path / "test_matrix.json"
    deferred_path  = output_path / "deferred_tests.json"
    run_list_path  = output_path / "tests_to_run.txt"

    with open(matrix_path, "w") as f:
        json.dump(matrix, f, indent=2)

    with open(deferred_path, "w") as f:
        json.dump({"tests": deferred_list, "schedule": "off-peak"}, f, indent=2)

    with open(run_list_path, "w") as f:
        f.write("\n".join(all_immediate))

    log.info("Matrix written   → %s (%d tests)", matrix_path, len(all_immediate))
    log.info("Deferred written → %s (%d tests)", deferred_path, len(deferred_list))
    log.info("Run list written → %s", run_list_path)

    return matrix_path, deferred_path, run_list_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — GITHUB CLI TRIGGER
# ─────────────────────────────────────────────────────────────────────────────

def trigger_via_gh_cli(
    repo:        str,
    workflow:    str,
    ref:         str = "main",
    inputs:      Optional[dict] = None,
) -> bool:
    """
    Trigger a GitHub Actions workflow via the GitHub CLI.
    Requires: gh auth login
    """
    cmd = ["gh", "workflow", "run", workflow, "--repo", repo, "--ref", ref]
    if inputs:
        for key, val in inputs.items():
            cmd += ["-f", f"{key}={val}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            log.info("Workflow triggered via gh CLI: %s", workflow)
            return True
        log.warning("gh CLI trigger failed: %s", result.stderr.strip())
    except FileNotFoundError:
        log.warning("GitHub CLI (gh) not installed — skipping CLI trigger")
    except subprocess.TimeoutExpired:
        log.warning("gh CLI trigger timed out")
    return False


def trigger_via_api(
    repo:     str,
    workflow: str,
    ref:      str = "main",
    inputs:   Optional[dict] = None,
) -> bool:
    """
    Trigger a GitHub Actions workflow via the REST API.
    Works without the gh CLI.
    """
    url  = f"{GITHUB_API_BASE}/repos/{repo}/actions/workflows/{workflow}/dispatches"
    body = {"ref": ref, "inputs": inputs or {}}
    resp = requests.post(url, headers=_headers(), json=body, timeout=30)
    if resp.status_code == 204:
        log.info("Workflow triggered via API: %s", workflow)
        return True
    log.warning("API trigger failed: %d %s", resp.status_code, resp.text[:200])
    return False


def set_actions_output(key: str, value: str):
    """
    Write a key=value pair to the GitHub Actions output file.
    Used to pass the test matrix to subsequent steps.
    """
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            # For multiline values, use the heredoc format
            if "\n" in str(value):
                delimiter = f"EOF_{int(time.time())}"
                f.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")
            else:
                f.write(f"{key}={value}\n")
        log.info("Actions output set: %s", key)
    else:
        # Not in Actions — print for debugging
        print(f"::set-output name={key}::{value}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run(
    schedule:          Optional[dict] = None,
    pruning_decision:  Optional[dict] = None,
    impact:            Optional[dict] = None,
    repo:              str = REPO_NAME,
    pr_number:         int = PR_NUMBER,
    trigger_workflow:  bool = True,
    workflow_file:     str = "greenops_run_tests.yml",
) -> dict:
    """
    Main entry point.

    Loads schedule/decision artifacts from disk if not passed directly.
    Posts PR comment, updates labels, writes matrix, triggers workflow.

    Returns summary dict.
    """
    log.info("=" * 60)
    log.info("GitHub Actions Runner")
    log.info("Repo: %s  PR: #%d", repo, pr_number)
    log.info("=" * 60)

    # ── Load artifacts from disk if not provided ──────────────────────────────
    if schedule is None:
        if SCHEDULE_PATH.exists():
            with open(SCHEDULE_PATH) as f:
                schedule = json.load(f)
            log.info("Loaded schedule from %s", SCHEDULE_PATH)
        else:
            log.warning("No schedule found at %s — using empty schedule", SCHEDULE_PATH)
            schedule = {"schedule_now": [], "schedule_deferred": [],
                        "historic_failure_tests": [], "provider": PROVIDER,
                        "selected_zone": "ap-south-1", "selected_city": "Mumbai",
                        "carbon_intensity": 659.0, "carbon_score": 0.732,
                        "total_ops_estimated": 0, "total_carbon_gco2": 0.0,
                        "recommendation": ""}

    if pruning_decision is None and DECISION_PATH.exists():
        with open(DECISION_PATH) as f:
            pruning_decision = json.load(f)
        log.info("Loaded pruning decision from %s", DECISION_PATH)
    pruning_decision = pruning_decision or {"prune": [], "pf_scores": {}, "pruning_rate": 0}

    if impact is None and IMPACT_PATH.exists():
        with open(IMPACT_PATH) as f:
            impact = json.load(f)

    # ── Step 1: Write test matrix for Actions ─────────────────────────────────
    log.info("Step 1: Writing test matrix ...")
    matrix_path, deferred_path, run_list_path = write_test_matrix(schedule, OUTPUT_DIR)

    # Load the matrix JSON so we can set it as an Actions output
    with open(matrix_path) as f:
        matrix_json = json.load(f)

    # ── Step 2: Create Green-Ops labels in repo ───────────────────────────────
    if repo and GITHUB_TOKEN:
        log.info("Step 2: Creating PR labels ...")
        create_pr_labels_if_missing(repo)

    # ── Step 3: Post PR comment ───────────────────────────────────────────────
    comment_posted = False
    if repo and pr_number and GITHUB_TOKEN:
        log.info("Step 3: Posting PR comment ...")
        comment_body  = build_pr_comment(schedule, pruning_decision, impact)
        comment_posted = post_pr_comment(repo, pr_number, comment_body)

        # Save comment locally regardless
        comment_path = OUTPUT_DIR / f"pr_comment_pr{pr_number}.md"
        comment_path.write_text(comment_body)
        log.info("Comment saved → %s", comment_path)
    else:
        log.info("Step 3: Skipping PR comment (no token/repo configured)")
        comment_body = build_pr_comment(schedule, pruning_decision, impact)
        comment_path = OUTPUT_DIR / "pr_comment_draft.md"
        comment_path.write_text(comment_body)
        log.info("Comment draft saved → %s", comment_path)

    # ── Step 4: Update PR labels ──────────────────────────────────────────────
    labels_applied = []
    if repo and pr_number and GITHUB_TOKEN:
        log.info("Step 4: Updating PR labels ...")
        labels = []
        if pruning_decision.get("prune"):
            labels.append("greenops:pruned")
        if schedule.get("schedule_deferred"):
            labels.append("greenops:deferred")
        if schedule.get("historic_failure_tests"):
            labels.append("greenops:historic")
        if schedule.get("carbon_score", 1.0) < 0.5:
            labels.append("greenops:clean")
        if labels:
            update_pr_labels(repo, pr_number, labels)
            labels_applied = labels

    # ── Step 5: Post commit status ────────────────────────────────────────────
    if repo and pr_number and GITHUB_TOKEN:
        log.info("Step 5: Posting commit status ...")
        sha = get_pr_head_sha(repo, pr_number)
        if sha:
            n_run      = len(schedule.get("schedule_now", []))
            n_historic = len(schedule.get("historic_failure_tests", []))
            n_pruned   = len(pruning_decision.get("prune", []))
            post_commit_status(
                repo        = repo,
                sha         = sha,
                state       = "pending",
                description = (
                    f"Green-Ops: running {n_run + n_historic} tests "
                    f"(pruned {n_pruned}) on "
                    f"{schedule.get('provider','').upper()}/"
                    f"{schedule.get('selected_zone','')} "
                    f"({schedule.get('carbon_intensity',0):.0f} gCO₂/kWh)"
                ),
            )

    # ── Step 6: Set GitHub Actions outputs ────────────────────────────────────
    log.info("Step 6: Setting Actions outputs ...")
    set_actions_output("matrix",        json.dumps(matrix_json))
    set_actions_output("test_count",    str(len(matrix_json.get("include", []))))
    set_actions_output("provider",      schedule.get("provider", PROVIDER))
    set_actions_output("zone",          schedule.get("selected_zone", "ap-south-1"))
    set_actions_output("carbon_score",  str(schedule.get("carbon_score", 0)))
    set_actions_output("pruning_rate",  str(pruning_decision.get("pruning_rate", 0)))

    # ── Step 7: Trigger workflow (optional) ───────────────────────────────────
    triggered = False
    if trigger_workflow and repo and GITHUB_TOKEN:
        log.info("Step 7: Triggering GitHub Actions workflow ...")
        pr_ref = f"refs/pull/{pr_number}/head" if pr_number else "main"
        inputs = {
            "test_matrix":  json.dumps(matrix_json),
            "provider":     schedule.get("provider", PROVIDER),
            "zone":         schedule.get("selected_zone", "ap-south-1"),
            "pr_number":    str(pr_number),
        }
        triggered = (
            trigger_via_gh_cli(repo, workflow_file, pr_ref, inputs) or
            trigger_via_api(repo, workflow_file, pr_ref, inputs)
        )
    else:
        log.info("Step 7: Workflow trigger skipped (set trigger_workflow=True to enable)")

    summary = {
        "tests_scheduled_now":    len(schedule.get("schedule_now", [])),
        "tests_historic":         len(schedule.get("historic_failure_tests", [])),
        "tests_deferred":         len(schedule.get("schedule_deferred", [])),
        "tests_pruned":           len(pruning_decision.get("prune", [])),
        "provider":               schedule.get("provider", PROVIDER),
        "zone":                   schedule.get("selected_zone", ""),
        "city":                   schedule.get("selected_city", ""),
        "carbon_intensity":       schedule.get("carbon_intensity", 0),
        "total_carbon_gco2":      schedule.get("total_carbon_gco2", 0),
        "matrix_path":            str(matrix_path),
        "deferred_path":          str(deferred_path),
        "run_list_path":          str(run_list_path),
        "comment_posted":         comment_posted,
        "labels_applied":         labels_applied,
        "workflow_triggered":     triggered,
    }

    _print_summary(summary, schedule)

    summary_path = OUTPUT_DIR / "runner_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Runner summary → %s", summary_path)

    return summary


def _print_summary(summary: dict, schedule: dict):
    print(f"\n{'='*65}")
    print("Green-Ops GitHub Actions Runner — Complete")
    print(f"{'='*65}")
    print(f"  Provider      : {summary['provider'].upper()}")
    print(f"  Zone          : {summary['zone']}  ({summary['city']})")
    print(f"  Carbon        : {summary['carbon_intensity']:.0f} gCO₂/kWh")
    print(f"  Total carbon  : {summary['total_carbon_gco2']:.6f} gCO₂")
    print()
    print(f"  ✅ Run now    : {summary['tests_scheduled_now']}")
    print(f"  ⚡ Historic   : {summary['tests_historic']}")
    print(f"  ⏸ Deferred   : {summary['tests_deferred']}")
    print(f"  🚫 Pruned     : {summary['tests_pruned']}")
    print()
    print(f"  Matrix file   : {summary['matrix_path']}")
    print(f"  Run list      : {summary['run_list_path']}")
    print(f"  PR comment    : {'✓ posted' if summary['comment_posted'] else '✗ skipped (no token)'}")
    print(f"  Labels        : {summary['labels_applied'] or 'skipped'}")
    print(f"  Workflow      : {'✓ triggered' if summary['workflow_triggered'] else '✗ skipped'}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GITHUB ACTIONS WORKFLOW YAML GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_workflow_yaml(output_path: Path):
    """
    Generate the complete GitHub Actions workflow YAML.
    Two workflows are generated:
      1. greenops_trigger.yml    — runs on PR, executes full pipeline
      2. greenops_run_tests.yml  — dispatched by the runner, executes test matrix
    """
    trigger_yml = """\
# .github/workflows/greenops_trigger.yml
# ==========================================
# Green-Ops: Runs on every PR to compute test selection + carbon schedule
# Then dispatches greenops_run_tests.yml with only the pruned test set.

name: Green-Ops Carbon-Aware CI

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents:      read
  pull-requests: write
  statuses:      write
  checks:        write

jobs:
  greenops-pipeline:
    name: Green-Ops Pipeline
    runs-on: ubuntu-latest

    outputs:
      matrix:        ${{ steps.runner.outputs.matrix }}
      test_count:    ${{ steps.runner.outputs.test_count }}
      provider:      ${{ steps.runner.outputs.provider }}
      zone:          ${{ steps.runner.outputs.zone }}
      carbon_score:  ${{ steps.runner.outputs.carbon_score }}
      pruning_rate:  ${{ steps.runner.outputs.pruning_rate }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Green-Ops dependencies
        run: |
          pip install pandas numpy scikit-learn xgboost requests scipy transformers torch

      - name: Run Green-Ops full pipeline
        id: pipeline
        env:
          GITHUB_TOKEN:   ${{ secrets.GITHUB_TOKEN }}
          REPO_NAME:      ${{ github.repository }}
          PR_NUMBER:      ${{ github.event.pull_request.number }}
          GREENOPS_OUTPUT: ./greenops_output
          GREENOPS_PROVIDER: aws
        run: |
          python github_ci_integration.py
          python -c "
          from graphcodebert_embeddings import GraphCodeBERTEmbedder
          from llm_impact_analyzer import LLMImpactAnalyzer
          from xgboost_gatekeeper import run_gatekeeper_pipeline
          from carbon_aware_scheduler import CarbonAwareScheduler
          import json, numpy as np
          # Load artifacts from CI integration step
          with open('./greenops_output/preprocessing_artifacts_pr${{ github.event.pull_request.number }}.json') as f:
              artifacts = json.load(f)
          # Build mock embeddings (replace with real GraphCodeBERT call)
          dim = 768
          changed_emb = {m['module_hash']: np.random.rand(dim).astype(np.float32)
                         for m in artifacts['modules'] if m.get('module_hash')}
          test_emb = {}  # populated from your test registry
          # Run gatekeeper
          decision = run_gatekeeper_pipeline(
              changed_modules=artifacts['modules'],
              module_registry=artifacts['modules'],
              changed_embeddings=changed_emb,
              test_embeddings=test_emb,
              change_size=sum(1 for _ in open('./greenops_output/raw_diff_pr${{ github.event.pull_request.number }}.diff')),
          )
          # Schedule
          scheduler = CarbonAwareScheduler(provider='aws')
          schedule  = scheduler.schedule(decision)
          print('Schedule complete')
          "

      - name: Post results to GitHub
        id: runner
        env:
          GITHUB_TOKEN:   ${{ secrets.GITHUB_TOKEN }}
          REPO_NAME:      ${{ github.repository }}
          PR_NUMBER:      ${{ github.event.pull_request.number }}
          GREENOPS_OUTPUT: ./greenops_output
          GREENOPS_PROVIDER: aws
        run: python github_actions_runner.py

      - name: Upload Green-Ops artifacts
        uses: actions/upload-artifact@v4
        with:
          name: greenops-artifacts-pr${{ github.event.pull_request.number }}
          path: |
            greenops_output/pruning_decision.json
            greenops_output/test_schedule.json
            greenops_output/impact_analysis.json
            greenops_output/test_matrix.json
            greenops_output/pr_comment_*.md
          retention-days: 30

  run-tests:
    name: Run Test — ${{ matrix.test }}
    needs: greenops-pipeline
    if: ${{ needs.greenops-pipeline.outputs.test_count > 0 }}
    runs-on: ubuntu-latest

    strategy:
      fail-fast: false
      matrix: ${{ fromJson(needs.greenops-pipeline.outputs.matrix) }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install test dependencies
        run: pip install -r requirements.txt

      - name: Run test — ${{ matrix.test }}
        id: run_test
        env:
          TEST_NAME:     ${{ matrix.test }}
          GREENOPS_ZONE: ${{ matrix.zone }}
          GREENOPS_PROVIDER: ${{ matrix.provider }}
        run: |
          echo "Running ${{ matrix.test }} on ${{ matrix.provider }}/${{ matrix.zone }}"
          # ── Replace with your actual test runner command ──────────────────
          # Examples:
          #   pytest tests/ -k "${{ matrix.test }}" -v
          #   mvn test -Dtest="${{ matrix.test }}"
          #   gradle test --tests "${{ matrix.test }}"
          python -m pytest tests/ -k "${{ matrix.test }}" -v \
            --tb=short \
            --junit-xml=test-results/${{ matrix.test }}.xml \
            || true

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ matrix.test }}
          path: test-results/

  post-final-status:
    name: Post Final Status
    needs: [greenops-pipeline, run-tests]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install requests

      - name: Post final commit status
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO_NAME:    ${{ github.repository }}
          PR_NUMBER:    ${{ github.event.pull_request.number }}
        run: |
          python -c "
          import os, requests
          token   = os.environ['GITHUB_TOKEN']
          repo    = os.environ['REPO_NAME']
          pr_num  = int(os.environ['PR_NUMBER'])
          headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github.v3+json'}
          pr_resp = requests.get(f'https://api.github.com/repos/{repo}/pulls/{pr_num}', headers=headers)
          sha     = pr_resp.json().get('head', {}).get('sha', '')
          if sha:
              requests.post(
                  f'https://api.github.com/repos/{repo}/statuses/{sha}',
                  headers=headers,
                  json={
                      'state':       'success',
                      'description': 'Green-Ops: Carbon-aware test run complete',
                      'context':     'Green-Ops/carbon',
                  }
              )
              print(f'Final status posted for {sha[:8]}')
          "
"""

    run_tests_yml = """\
# .github/workflows/greenops_run_tests.yml
# ==========================================
# Green-Ops: Dispatched by greenops_trigger.yml with the pruned test matrix.
# Can also be triggered manually via: gh workflow run greenops_run_tests.yml

name: Green-Ops Run Pruned Tests

on:
  workflow_dispatch:
    inputs:
      test_matrix:
        description: "JSON matrix from Green-Ops scheduler"
        required: true
        default: '{"include": []}'
      provider:
        description: "Cloud provider (aws/azure/gcp)"
        required: false
        default: "aws"
      zone:
        description: "Datacenter zone (e.g. ap-south-1)"
        required: false
        default: "ap-south-1"
      pr_number:
        description: "PR number for status updates"
        required: false
        default: "0"

jobs:
  run-pruned-tests:
    name: Run — ${{ matrix.test }}
    runs-on: ubuntu-latest

    strategy:
      fail-fast: false
      matrix: ${{ fromJson(github.event.inputs.test_matrix) }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up environment
        run: |
          echo "GREENOPS_PROVIDER=${{ github.event.inputs.provider }}" >> $GITHUB_ENV
          echo "GREENOPS_ZONE=${{ github.event.inputs.zone }}"         >> $GITHUB_ENV
          echo "Running ${{ matrix.test }} on ${{ github.event.inputs.provider }}/${{ github.event.inputs.zone }}"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run test — ${{ matrix.test }}
        run: |
          # ── Replace with your test runner ────────────────────────────────
          python -m pytest tests/ -k "${{ matrix.test }}" -v \
            --tb=short \
            --junit-xml=test-results/${{ matrix.test }}.xml \
            || true

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ matrix.test }}
          path: test-results/

  run-deferred-tests:
    name: Run Deferred Tests (Off-Peak)
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Download deferred test list
        uses: actions/download-artifact@v4
        with:
          name: greenops-artifacts-pr${{ github.event.inputs.pr_number }}
          path: greenops_output/

      - name: Run deferred tests
        run: |
          if [ -f greenops_output/deferred_tests.json ]; then
            python -c "
          import json, subprocess
          with open('greenops_output/deferred_tests.json') as f:
              data = json.load(f)
          for test in data.get('tests', []):
              print(f'Running deferred test: {test}')
              subprocess.run(['python', '-m', 'pytest', 'tests/', '-k', test, '-v'], check=False)
            "
          fi
"""

    # Write YAMLs to .github/workflows/
    workflows_dir = Path(".github/workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)

    trigger_path    = workflows_dir / "greenops_trigger.yml"
    run_tests_path  = workflows_dir / "greenops_run_tests.yml"

    trigger_path.write_text(trigger_yml)
    run_tests_path.write_text(run_tests_yml)

    # Also copy to output dir for inspection
    (output_path / "greenops_trigger.yml").write_text(trigger_yml)
    (output_path / "greenops_run_tests.yml").write_text(run_tests_yml)

    log.info("Workflows written:")
    log.info("  %s", trigger_path)
    log.info("  %s", run_tests_path)

    return str(trigger_path), str(run_tests_path)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Green-Ops GitHub Actions Runner")
    parser.add_argument("--generate-workflows", action="store_true",
                        help="Generate GitHub Actions YAML workflow files")
    parser.add_argument("--no-trigger", action="store_true",
                        help="Skip workflow trigger (dry run)")
    args = parser.parse_args()

    if args.generate_workflows:
        t_path, r_path = generate_workflow_yaml(OUTPUT_DIR)
        print(f"\nWorkflows generated:")
        print(f"  {t_path}")
        print(f"  {r_path}")
        print("\nCommit these to your repo:")
        print("  git add .github/workflows/greenops_trigger.yml")
        print("  git add .github/workflows/greenops_run_tests.yml")
        print("  git commit -m 'ci: add Green-Ops carbon-aware workflows'")
        print("  git push")
    else:
        summary = run(trigger_workflow=not args.no_trigger)

        print(f"\nNext steps:")
        print(f"  1. Commit matrix to trigger test run:")
        print(f"       gh workflow run greenops_run_tests.yml \\")
        print(f"         -f test_matrix='$(cat {summary['matrix_path']}')")
        print(f"  2. View PR comment draft:")
        print(f"       cat {OUTPUT_DIR}/pr_comment_draft.md")
        print(f"  3. Generate GitHub Actions workflows:")
        print(f"       python github_actions_runner.py --generate-workflows")
