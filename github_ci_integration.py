"""
github_ci_integration.py
=========================
Green-Ops Framework — GitHub CLI / Actions / Diff Integration

Connects the Input Layer (Git PR Code Diff) to the Preprocessing Layer
(AST Parser + Module Hash Generator) as shown in the architecture diagram.

Flow:
  1. GitHub Actions webhook / CLI triggers this on a new PR event
  2. Fetch the raw git diff for the PR (via GitHub CLI or API)
  3. Pass each changed file's content to ast_parser.py (AST parsing)
  4. Pass parsed AST output to module_db.py (module hash generation)
  5. Emit structured JSON artifacts for the Semantic Layer (GraphCodeBERT)

BRANCHING NOTE:
  ast_parser.py  comes from branch: preprocessing embeddings
  module_db.py   comes from branch: feature/module-hash-generator
  After both branches merge to main, these imports just work — no branch
  references are needed in runtime code. If you are running this BEFORE
  the merge, set PYTHONPATH to point at the checked-out branch directories
  (see RUN INSTRUCTIONS at the bottom of this file).

Dependencies:
    pip install PyGithub requests

Run locally (pre-merge, pointing at branch checkouts):
    export GITHUB_TOKEN=ghp_...
    export PR_NUMBER=123
    export REPO_NAME=your-org/your-repo
    python github_ci_integration.py

Run inside GitHub Actions:
    See .github/workflows/greenops_trigger.yml (generated alongside this file)
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("greenops.ci")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — all values read from environment variables so this works
# both locally and inside GitHub Actions without code changes
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN     = os.environ.get("GITHUB_TOKEN", "")
REPO_NAME        = os.environ.get("REPO_NAME", "")          # e.g. "apache/kafka"
PR_NUMBER        = int(os.environ.get("PR_NUMBER", "0"))
OUTPUT_DIR       = Path(os.environ.get("GREENOPS_OUTPUT", "./greenops_output"))

# Paths to your module files — defaults assume they are in the same directory.
# Override with env vars if your repo layout differs.
AST_PARSER_PATH  = Path(os.environ.get("AST_PARSER_PATH",  "./ast_parser.py"))
MODULE_DB_PATH   = Path(os.environ.get("MODULE_DB_PATH",   "./module_db.py"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DYNAMIC MODULE LOADER
# Loads ast_parser.py and module_db.py at runtime by file path.
# This is branch-agnostic: it works whether the files came from a feature
# branch or main — you just point the path at wherever the file lives.
# ─────────────────────────────────────────────────────────────────────────────

def load_module_from_path(module_name: str, file_path: Path):
    """
    Dynamically load a Python module from an absolute file path.
    This lets us import ast_parser.py and module_db.py regardless of
    which branch they were merged from or where they sit in the repo.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find {module_name} at {file_path}\n"
            f"Set the environment variable {module_name.upper().replace('.','_')}_PATH "
            f"to the correct path, or ensure the file is in the current directory."
        )
    spec   = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    log.info("Loaded module '%s' from %s", module_name, file_path)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — GIT DIFF FETCHER
# Supports three modes:
#   A. GitHub Actions environment (GITHUB_TOKEN + PR event context)
#   B. GitHub CLI (gh pr diff) — for local development
#   C. GitHub REST API — fallback, always works with a valid token
# ─────────────────────────────────────────────────────────────────────────────

def fetch_diff_via_gh_cli(repo: str, pr_number: int) -> Optional[str]:
    """
    Fetch PR diff using the GitHub CLI (gh).
    Requires: brew install gh && gh auth login
    """
    try:
        result = subprocess.run(
            ["gh", "pr", "diff", str(pr_number), "--repo", repo],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            log.info("Fetched diff via GitHub CLI (%d chars)", len(result.stdout))
            return result.stdout
        log.warning("gh cli failed: %s", result.stderr.strip())
    except FileNotFoundError:
        log.warning("GitHub CLI (gh) not installed — falling back to REST API")
    except subprocess.TimeoutExpired:
        log.warning("gh cli timed out")
    return None


def fetch_diff_via_api(repo: str, pr_number: int, token: str) -> Optional[str]:
    """
    Fetch PR diff using the GitHub REST API.
    Works in GitHub Actions with the automatic GITHUB_TOKEN secret.
    """
    url     = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github.v3.diff",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            log.info("Fetched diff via REST API (%d chars)", len(resp.text))
            return resp.text
        log.error("API returned %d: %s", resp.status_code, resp.text[:200])
    except requests.RequestException as exc:
        log.error("REST API request failed: %s", exc)
    return None


def fetch_diff_from_actions_event() -> Optional[str]:
    """
    Inside GitHub Actions, the PR diff can be obtained from the local
    git history because Actions checks out the merge commit.
    """
    try:
        # Get the base SHA from the Actions event payload
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        base_sha   = ""
        if event_path and Path(event_path).exists():
            with open(event_path) as f:
                event = json.load(f)
            base_sha = (event.get("pull_request", {})
                             .get("base", {})
                             .get("sha", ""))

        if not base_sha:
            # Fallback: diff against the previous commit
            base_sha = "HEAD~1"

        result = subprocess.run(
            ["git", "diff", base_sha, "HEAD"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout:
            log.info("Fetched diff from git (Actions context, %d chars)",
                     len(result.stdout))
            return result.stdout
    except Exception as exc:
        log.warning("Actions git diff failed: %s", exc)
    return None


def fetch_pr_diff(repo: str, pr_number: int, token: str) -> str:
    """
    Master diff fetcher — tries all three methods in order.
    Returns the raw unified diff string.
    """
    # Method 1: Inside GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true":
        diff = fetch_diff_from_actions_event()
        if diff:
            return diff

    # Method 2: GitHub CLI (local dev)
    diff = fetch_diff_via_gh_cli(repo, pr_number)
    if diff:
        return diff

    # Method 3: REST API (always works if token is valid)
    diff = fetch_diff_via_api(repo, pr_number, token)
    if diff:
        return diff

    raise RuntimeError(
        f"Could not fetch diff for PR #{pr_number} in {repo}. "
        "Check GITHUB_TOKEN, REPO_NAME, and PR_NUMBER environment variables."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DIFF → FILE EXTRACTOR
# Splits the unified diff into per-file source code chunks and writes
# them to a temp directory so ast_parser.py can parse them as real files.
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".py", ".java", ".ts", ".js", ".go", ".kt", ".scala",
                         ".cpp", ".c", ".cs", ".rb", ".rs"}


def extract_files_from_diff(diff_text: str, work_dir: Path) -> list[dict]:
    """
    Parse the unified diff and reconstruct the ADDED/MODIFIED lines of each
    changed file. Writes reconstructed files to work_dir for AST parsing.

    Returns a list of dicts with keys:
        filepath, language, added_lines, removed_lines, file_path_on_disk
    """
    files   = []
    current = None
    lines   = diff_text.splitlines()

    for line in lines:
        # New file section in unified diff
        if line.startswith("diff --git"):
            if current and current["added_lines"]:
                _write_file(current, work_dir)
                files.append(current)
            current = None

        elif line.startswith("+++ b/"):
            filepath = line[6:].strip()
            ext      = Path(filepath).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                current = None
                continue
            current = {
                "filepath":         filepath,
                "language":         _detect_language(filepath),
                "added_lines":      [],
                "removed_lines":    [],
                "file_path_on_disk": None,
            }

        elif current is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current["added_lines"].append(line[1:])   # strip leading +
            elif line.startswith("-") and not line.startswith("---"):
                current["removed_lines"].append(line[1:]) # strip leading -

    # Flush last file
    if current and current["added_lines"]:
        _write_file(current, work_dir)
        files.append(current)

    log.info("Extracted %d parseable source files from diff", len(files))
    return files


def _write_file(file_info: dict, work_dir: Path):
    """Write reconstructed file content to disk for AST parsing."""
    safe_name = file_info["filepath"].replace("/", "_")
    out_path  = work_dir / safe_name
    out_path.write_text("\n".join(file_info["added_lines"]), encoding="utf-8")
    file_info["file_path_on_disk"] = str(out_path)


def _detect_language(filepath: str) -> str:
    ext_map = {
        ".py": "python", ".java": "java", ".ts": "typescript",
        ".js": "javascript", ".go": "go", ".kt": "kotlin",
        ".scala": "scala", ".cpp": "cpp", ".c": "c",
        ".cs": "csharp", ".rb": "ruby", ".rs": "rust",
    }
    return ext_map.get(Path(filepath).suffix.lower(), "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — AST PARSER INTEGRATION
# Calls ast_parser.py (from preprocessing embeddings branch) on each file.
# Expects ast_parser.py to expose: parse_file(filepath) -> dict
# ─────────────────────────────────────────────────────────────────────────────

def run_ast_parser(ast_parser_module, changed_files: list[dict]) -> list[dict]:
    """
    Call ast_parser.parse_file() on each changed source file.

    Expected interface for ast_parser.py:
        def parse_file(filepath: str) -> dict:
            # Returns a dict with keys like:
            # {
            #   "functions": [...],
            #   "classes": [...],
            #   "imports": [...],
            #   "tokens": [...],
            #   "ast_hash": "...",   # if your ast_parser generates one
            # }

    If your ast_parser.py has a different function name or signature,
    update the call below to match.
    """
    results = []

    # Detect the correct entry point in ast_parser.py
    parse_fn = None
    for candidate in ["parse_file", "parse", "run", "analyze"]:
        if hasattr(ast_parser_module, candidate):
            parse_fn = getattr(ast_parser_module, candidate)
            log.info("Using ast_parser.%s() as entry point", candidate)
            break

    if parse_fn is None:
        log.error(
            "ast_parser.py does not expose parse_file(), parse(), run(), or analyze(). "
            "Please check the function name and update this integration."
        )
        return results

    for f in changed_files:
        disk_path = f.get("file_path_on_disk")
        if not disk_path or not Path(disk_path).exists():
            log.warning("Skipping %s — file not written to disk", f["filepath"])
            continue
        try:
            ast_result = parse_fn(disk_path)
            results.append({
                "filepath":  f["filepath"],
                "language":  f["language"],
                "ast_result": ast_result,
            })
            log.info("AST parsed: %s", f["filepath"])
        except Exception as exc:
            log.warning("AST parse failed for %s: %s", f["filepath"], exc)

    log.info("AST parsing complete: %d / %d files succeeded",
             len(results), len(changed_files))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MODULE HASH GENERATOR INTEGRATION
# Calls module_db.py (from feature/module-hash-generator branch).
# Expects module_db.py to expose: generate_hash(ast_result) -> str
#   and optionally: store_module(module_info) -> None
# ─────────────────────────────────────────────────────────────────────────────

def run_module_hash_generator(
    module_db_module,
    ast_results: list[dict],
    pr_number: int,
    repo: str,
) -> list[dict]:
    """
    Call module_db.generate_hash() on each AST result to produce a stable
    content-addressable hash for each changed module.

    Expected interface for module_db.py:
        def generate_hash(ast_result: dict) -> str:
            # Returns a hex string hash of the module's semantic content

        def store_module(module_info: dict) -> None:  # optional
            # Persists module info to PostgreSQL / S3 (as in your architecture)

    If your module_db.py has a different function name, update below.
    """
    enriched = []

    # Detect entry points
    hash_fn  = None
    store_fn = None
    for candidate in ["generate_hash", "compute_hash", "hash_module", "get_hash"]:
        if hasattr(module_db_module, candidate):
            hash_fn = getattr(module_db_module, candidate)
            log.info("Using module_db.%s() for hashing", candidate)
            break

    for candidate in ["store_module", "save_module", "persist", "insert"]:
        if hasattr(module_db_module, candidate):
            store_fn = getattr(module_db_module, candidate)
            log.info("Using module_db.%s() for storage", candidate)
            break

    if hash_fn is None:
        log.warning(
            "module_db.py does not expose generate_hash() or similar. "
            "Hash generation will be skipped. "
            "Check the function name and update this integration."
        )

    for item in ast_results:
        module_hash = None

        if hash_fn:
            try:
                module_hash = hash_fn(item["ast_result"])
            except Exception as exc:
                log.warning("Hash generation failed for %s: %s",
                            item["filepath"], exc)

        module_info = {
            "repo":         repo,
            "pr_number":    pr_number,
            "filepath":     item["filepath"],
            "language":     item["language"],
            "module_hash":  module_hash,
            "ast_result":   item["ast_result"],
        }

        # Persist to DB/S3 if store function is available
        if store_fn:
            try:
                store_fn(module_info)
                log.info("Stored module: %s (hash=%s)",
                         item["filepath"], module_hash)
            except Exception as exc:
                log.warning("Storage failed for %s: %s", item["filepath"], exc)

        enriched.append(module_info)

    log.info("Module hash generation complete: %d modules processed", len(enriched))
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ARTIFACT WRITER
# Writes JSON artifacts consumed by the Semantic Layer (GraphCodeBERT)
# and the Generative AI Agent (dependency graph synthesis).
# ─────────────────────────────────────────────────────────────────────────────

def write_artifacts(
    pr_number: int,
    repo: str,
    module_records: list[dict],
    diff_text: str,
):
    """
    Write structured JSON artifacts that downstream layers consume:
      - preprocessing_artifacts.json  → input to GraphCodeBERT embedding
      - diff_summary.json             → metadata for the Generative AI Agent
    """
    # Preprocessing artifact (Semantic Layer input)
    preprocessing_artifact = {
        "pr_number":   pr_number,
        "repo":        repo,
        "module_count": len(module_records),
        "modules": [
            {
                "filepath":    m["filepath"],
                "language":    m["language"],
                "module_hash": m["module_hash"],
                # Include AST tokens if present — these feed GraphCodeBERT
                "tokens": (
                    m["ast_result"].get("tokens", [])
                    if isinstance(m.get("ast_result"), dict) else []
                ),
                "functions": (
                    m["ast_result"].get("functions", [])
                    if isinstance(m.get("ast_result"), dict) else []
                ),
                "imports": (
                    m["ast_result"].get("imports", [])
                    if isinstance(m.get("ast_result"), dict) else []
                ),
            }
            for m in module_records
        ],
    }

    # Diff summary artifact (Generative AI Agent input)
    diff_summary = {
        "pr_number":        pr_number,
        "repo":             repo,
        "total_diff_chars": len(diff_text),
        "changed_files":    [m["filepath"] for m in module_records],
        "languages":        list({m["language"] for m in module_records}),
        "module_hashes":    {
            m["filepath"]: m["module_hash"] for m in module_records
        },
    }

    preproc_path = OUTPUT_DIR / f"preprocessing_artifacts_pr{pr_number}.json"
    diff_path    = OUTPUT_DIR / f"diff_summary_pr{pr_number}.json"

    with open(preproc_path, "w") as f:
        json.dump(preprocessing_artifact, f, indent=2, default=str)

    with open(diff_path, "w") as f:
        json.dump(diff_summary, f, indent=2, default=str)

    log.info("Artifacts written:")
    log.info("  %s", preproc_path)
    log.info("  %s", diff_path)

    return str(preproc_path), str(diff_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Validate config ───────────────────────────────────────────────────────
    if not GITHUB_TOKEN:
        raise EnvironmentError("GITHUB_TOKEN is not set.")
    if not REPO_NAME:
        raise EnvironmentError("REPO_NAME is not set (e.g. export REPO_NAME=apache/kafka).")
    if PR_NUMBER == 0:
        raise EnvironmentError("PR_NUMBER is not set (e.g. export PR_NUMBER=1234).")

    log.info("=== Green-Ops GitHub CI Integration ===")
    log.info("Repo: %s  PR: #%d", REPO_NAME, PR_NUMBER)

    # ── Load modules from file paths (branch-agnostic) ────────────────────────
    ast_parser_module = load_module_from_path("ast_parser", AST_PARSER_PATH)
    module_db_module  = load_module_from_path("module_db",  MODULE_DB_PATH)

    # ── Fetch PR diff ─────────────────────────────────────────────────────────
    log.info("Fetching PR diff ...")
    diff_text = fetch_pr_diff(REPO_NAME, PR_NUMBER, GITHUB_TOKEN)

    # Save raw diff for audit trail
    raw_diff_path = OUTPUT_DIR / f"raw_diff_pr{PR_NUMBER}.diff"
    raw_diff_path.write_text(diff_text, encoding="utf-8")
    log.info("Raw diff saved -> %s", raw_diff_path)

    # ── Extract changed files from diff ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir      = Path(tmp_dir)
        changed_files = extract_files_from_diff(diff_text, work_dir)

        if not changed_files:
            log.info("No parseable source files changed in this PR. Exiting.")
            return

        # ── AST Parsing (ast_parser.py) ───────────────────────────────────────
        log.info("Running AST parser on %d files ...", len(changed_files))
        ast_results = run_ast_parser(ast_parser_module, changed_files)

        # ── Module Hash Generation (module_db.py) ─────────────────────────────
        log.info("Running module hash generator on %d AST results ...",
                 len(ast_results))
        module_records = run_module_hash_generator(
            module_db_module, ast_results, PR_NUMBER, REPO_NAME
        )

    # ── Write artifacts for downstream layers ─────────────────────────────────
    preproc_path, diff_path = write_artifacts(
        PR_NUMBER, REPO_NAME, module_records, diff_text
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GREEN-OPS PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  PR                 : #{PR_NUMBER} in {REPO_NAME}")
    print(f"  Files parsed       : {len(module_records)}")
    print(f"  Modules with hash  : {sum(1 for m in module_records if m['module_hash'])}")
    print(f"  Preprocessing JSON : {preproc_path}")
    print(f"  Diff summary JSON  : {diff_path}")
    print("\nNext step: pass preprocessing_artifacts.json to GraphCodeBERT embedder")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# RUN INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# SCENARIO A — Both branches already merged to main:
#   export GITHUB_TOKEN=ghp_...
#   export REPO_NAME=your-org/your-repo
#   export PR_NUMBER=123
#   python github_ci_integration.py
#
# SCENARIO B — Branches NOT yet merged (pre-merge testing):
#   # Check out both branches into separate directories, then:
#   export AST_PARSER_PATH=/path/to/preprocessing-embeddings-checkout/ast_parser.py
#   export MODULE_DB_PATH=/path/to/feature-module-hash-generator-checkout/module_db.py
#   export GITHUB_TOKEN=ghp_...
#   export REPO_NAME=your-org/your-repo
#   export PR_NUMBER=123
#   python github_ci_integration.py
#
# SCENARIO C — Inside GitHub Actions (see greenops_trigger.yml):
#   All env vars are injected by the workflow. No manual steps needed.
