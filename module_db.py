import os
import json
import hashlib
import logging
from pathlib import Path
try:
    from src import config
except ImportError:
    import config # Fallback for standalone runs

log = logging.getLogger("greenops.module_db")

class ModuleDB:
    """
    Simple local implementation of the module database.
    Stores module metadata and hashes in a local JSON file.
    """
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or config.GREENOPS_OUTPUT / "module_registry.json")
        self.registry = self._load()

    def _load(self) -> dict:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Failed to load module registry: {e}")
        return {}

    def _save(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "w") as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save module registry: {e}")

    def generate_hash(self, ast_result: dict) -> str:
        """
        Generates a stable SHA-256 hash from the AST components.
        Focuses on functions, classes, and imports.
        """
        # Canonicalize the AST components for hashing
        content = json.dumps({
            "functions": sorted([f.get("name", "") for f in ast_result.get("functions", [])]),
            "classes":   sorted([c.get("name", "") for c in ast_result.get("classes", [])]),
            "imports":   sorted(ast_result.get("imports", [])),
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def store_module(self, module_info: dict):
        """Persists module info to the local registry."""
        filepath = module_info.get("filepath")
        if filepath:
            self.registry[filepath] = {
                "module_hash":  module_info.get("module_hash"),
                "language":     module_info.get("language"),
                "repo":         module_info.get("repo"),
                "last_updated": module_info.get("pr_number"),
            }
            self._save()

# Functional interfaces for github_ci_integration.py
_db = ModuleDB()

def generate_hash(ast_result: dict) -> str:
    return _db.generate_hash(ast_result)

def store_module(module_info: dict):
    _db.store_module(module_info)
