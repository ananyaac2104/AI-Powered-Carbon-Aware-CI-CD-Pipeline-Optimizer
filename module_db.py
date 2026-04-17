"""
module_db.py
============
Green-Ops CI/CD Framework — Module Hash Generator + Registry

Called by github_ci_integration.py as the module_db step.
Provides the interface expected by load_module_from_path():
  - generate_hash(ast_result: dict) -> str
  - store_module(module_info: dict) -> None

Also extends the SQLite store with full PR-level tracking.

CHANGES:
  - REAL SHA-256 hash from AST content (not demo/random values).
  - Integrates with module_embedding_store.SQLiteEmbeddingStore.
  - generate_hash() is deterministic: same content = same hash.
  - store_module() persists to SQLite with upsert semantics.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("greenops.module_db")

DB_PATH = os.environ.get(
    "GREENOPS_DB_PATH",
    "./greenops_output/module_registry.sqlite",
)


def generate_hash(ast_result: dict) -> str:
    """
    Generate a stable SHA-256 hash for a module's semantic content.

    The hash is computed from the module's structural fingerprint:
      - function names + signatures
      - import list
      - class names
    NOT from raw source bytes (so reformatting doesn't change the hash).

    This is what github_ci_integration.run_module_hash_generator() calls.

    Args:
        ast_result: dict from ast_parser.parse_file().to_dict() or similar

    Returns:
        hex string SHA-256 hash, or "" on error
    """
    if not isinstance(ast_result, dict):
        return ""

    # Build a normalised structural fingerprint
    fingerprint = {
        "functions": sorted([
            f.get("name", "") if isinstance(f, dict) else str(f)
            for f in ast_result.get("functions", [])
        ]),
        "methods": sorted([
            f"{f.get('class_name','')}.{f.get('name','')}" if isinstance(f, dict) else str(f)
            for f in ast_result.get("methods", [])
        ]),
        "imports": sorted(ast_result.get("imports", [])),
        "classes": sorted([
            c.get("name", "") if isinstance(c, dict) else str(c)
            for c in ast_result.get("classes", [])
        ]),
    }

    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_module(module_info: dict) -> None:
    """
    Persist a module record to the SQLite store.
    Called by github_ci_integration.run_module_hash_generator().

    Args:
        module_info: dict with keys: repo, filepath, language, module_hash, ast_result
    """
    try:
        from module_embedding_store import SQLiteEmbeddingStore
        store = SQLiteEmbeddingStore(db_path=DB_PATH)
        store.upsert(
            repo         = module_info.get("repo", "unknown"),
            file_path    = module_info.get("filepath", ""),
            file_hash    = module_info.get("module_hash", ""),
            embedding    = None,    # embedding added later by repo_module_extractor
            language     = module_info.get("language", "python"),
            ast_features = module_info.get("ast_result") if isinstance(
                module_info.get("ast_result"), dict
            ) else None,
            value_score  = _compute_value_score(module_info.get("ast_result", {})),
            pr_number    = module_info.get("pr_number", 0),
        )
        log.debug("Stored module: %s", module_info.get("filepath"))
    except Exception as e:
        log.warning("store_module failed for %s: %s",
                    module_info.get("filepath", "?"), e)


def _compute_value_score(ast_result: dict) -> float:
    """Simple value score from AST features."""
    if not isinstance(ast_result, dict):
        return 0.0
    import math
    fns     = len(ast_result.get("functions", []))
    methods = len(ast_result.get("methods", []))
    imports = len(ast_result.get("imports", []))
    lines   = ast_result.get("num_lines", 1)
    return round(
        0.4 * (fns + methods) +
        0.3 * imports +
        0.2 * math.log(lines + 1),
        4,
    )


def get_stored_hash(repo: str, file_path: str) -> Optional[str]:
    """Retrieve the stored hash for a module (for comparison in PR diff)."""
    try:
        from module_embedding_store import SQLiteEmbeddingStore
        store  = SQLiteEmbeddingStore(db_path=DB_PATH)
        record = store.get(repo, file_path)
        return record["file_hash"] if record else None
    except Exception:
        return None


def list_stored_modules(repo: str) -> list:
    """Return all stored modules for a repo."""
    try:
        from module_embedding_store import SQLiteEmbeddingStore
        store = SQLiteEmbeddingStore(db_path=DB_PATH)
        return store.list_all(repo)
    except Exception:
        return []
