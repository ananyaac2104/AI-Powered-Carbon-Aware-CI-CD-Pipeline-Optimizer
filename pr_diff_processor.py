"""
pr_diff_processor.py
====================
Green-Ops CI/CD Framework — PR-Based Change Detection + Similarity Scoring

Detects changed files from git diff, recomputes embeddings ONLY for changed
modules, then compares against stored embeddings using cosine similarity.

Outputs:
  - changed_modules:     list of file_path strings
  - similarity_scores:   {file_path: cosine_similarity_to_stored}
  - hash_deltas:         {file_path: {old_hash, new_hash, changed: bool}}
  - structural_diffs:    {file_path: ASTDiff result dict}
  - impacted_tests:      test files transitively impacted

This module sits between github_ci_integration.py (diff fetch) and
the dependency graph builder + XGBoost gatekeeper.

USAGE:
    from pr_diff_processor import PRDiffProcessor
    processor = PRDiffProcessor(repo="org/repo", repo_root="/path/to/repo")
    result = processor.process_diff(diff_text="...", pr_number=42)
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from module_embedding_store import SQLiteEmbeddingStore, get_store, compute_file_hash
from repo_module_extractor import RepoModuleExtractor, get_embedder, EMBEDDING_DIM

log = logging.getLogger("greenops.pr_diff_processor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD   = float(os.environ.get("GREENOPS_SIM_THRESHOLD", "0.70"))
HASH_CHANGE_WEIGHT     = float(os.environ.get("GREENOPS_HASH_WEIGHT", "0.3"))
EMBEDDING_CHANGE_WEIGHT = float(os.environ.get("GREENOPS_EMB_WEIGHT", "0.7"))


# ─────────────────────────────────────────────────────────────────────────────
# DIFF PARSER
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rb", ".rs"}


def parse_changed_files_from_diff(diff_text: str) -> List[Dict]:
    """
    Parse a unified diff string and extract changed file info.

    Returns list of dicts:
      {filepath, added_lines, removed_lines, added_code, removed_code, language}
    """
    files:   List[Dict] = []
    current: Optional[Dict] = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            if current:
                files.append(current)
            current = None

        elif line.startswith("+++ b/"):
            filepath = line[6:].strip()
            ext      = Path(filepath).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                current = None
                continue
            lang_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".java": "java", ".go": "go", ".rb": "ruby", ".rs": "rust",
            }
            current = {
                "filepath":      filepath,
                "language":      lang_map.get(ext, "unknown"),
                "added_lines":   [],
                "removed_lines": [],
            }

        elif current is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current["added_lines"].append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current["removed_lines"].append(line[1:])

    if current:
        files.append(current)

    for f in files:
        f["added_code"]   = "\n".join(f["added_lines"])
        f["removed_code"] = "\n".join(f["removed_lines"])
        f["net_lines_changed"] = len(f["added_lines"]) + len(f["removed_lines"])

    log.info("Parsed %d changed source files from diff", len(files))
    return files


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingSimilarityEngine:
    """
    Computes cosine similarity between new PR embeddings and stored embeddings.
    A low similarity = significant semantic change → more tests needed.
    A high similarity = minor change → fewer tests needed.
    """

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        a = a.astype(np.float32).flatten()
        b = b.astype(np.float32).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_sim(
        query:  np.ndarray,        # shape [D]
        corpus: np.ndarray,        # shape [N, D]
    ) -> np.ndarray:               # shape [N]
        return cosine_similarity(
            query.reshape(1, -1).astype(np.float32),
            corpus.astype(np.float32),
        ).flatten()

    def compare_pr_to_stored(
        self,
        store:           SQLiteEmbeddingStore,
        repo:            str,
        pr_embeddings:   Dict[str, np.ndarray],  # {file_path: new_embedding}
    ) -> Dict[str, float]:
        """
        For each changed file, compute cosine similarity between its
        NEW embedding and its STORED (pre-PR) embedding.

        Interpretation:
          similarity ≈ 1.0  → tiny semantic change (likely safe to prune tests)
          similarity ≈ 0.5  → moderate change (run some tests)
          similarity ≈ 0.0  → massive semantic change (run all related tests)

        Returns: {file_path: similarity_score}
        """
        scores: Dict[str, float] = {}

        for file_path, new_emb in pr_embeddings.items():
            record = store.get(repo, file_path)
            if record is None or record.get("embedding") is None:
                # No stored embedding — brand new file → treat as max change
                scores[file_path] = 0.0
                log.info("New file (no stored embedding): %s → sim=0.0", file_path)
                continue

            stored_emb = record["embedding"]
            sim = self.cosine_sim(stored_emb, new_emb)
            scores[file_path] = round(float(sim), 6)
            log.info("Similarity %s: %.4f", file_path, sim)

        return scores

    def compare_changed_to_test_modules(
        self,
        store:             SQLiteEmbeddingStore,
        repo:              str,
        pr_embeddings:     Dict[str, np.ndarray],  # {source_file: embedding}
        test_files:        List[str],               # repo-relative test file paths
        top_k_per_change:  int = 20,
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute similarity between each CHANGED source file and every TEST file.
        This is the core of test selection: which tests are most semantically
        similar to what changed?

        Returns: {(source_file, test_file): similarity_score}
        """
        result: Dict[Tuple[str, str], float] = {}

        if not test_files:
            log.warning("No test files provided for similarity comparison")
            return result

        # Load test embeddings from store
        test_embeddings: Dict[str, np.ndarray] = {}
        for tf in test_files:
            record = store.get(repo, tf)
            if record and record.get("embedding") is not None:
                test_embeddings[tf] = record["embedding"]

        if not test_embeddings:
            log.warning("No test embeddings found in store for %d test files", len(test_files))
            return result

        test_paths  = list(test_embeddings.keys())
        test_matrix = np.vstack(list(test_embeddings.values()))  # [T, D]

        for source_file, src_emb in pr_embeddings.items():
            sims = self.batch_cosine_sim(src_emb, test_matrix)  # [T]
            # Get top_k most similar tests
            top_indices = np.argsort(sims)[::-1][:top_k_per_change]
            for idx in top_indices:
                sim = float(sims[idx])
                if sim >= SIMILARITY_THRESHOLD:
                    result[(source_file, test_paths[idx])] = round(sim, 6)

        log.info(
            "Test similarity computed: %d source→test pairs above threshold %.2f",
            len(result), SIMILARITY_THRESHOLD,
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# HASH DELTA TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def compute_hash_deltas(
    store:         SQLiteEmbeddingStore,
    repo:          str,
    changed_files: List[Dict],  # from parse_changed_files_from_diff()
    repo_root:     str,
) -> Dict[str, Dict]:
    """
    Compare current file hashes against stored hashes.
    Returns: {file_path: {old_hash, new_hash, changed, is_new}}
    """
    deltas: Dict[str, Dict] = {}

    for f in changed_files:
        fp       = f["filepath"]
        abs_path = Path(repo_root) / fp
        new_hash = compute_file_hash(str(abs_path)) if abs_path.exists() else ""

        record   = store.get(repo, fp)
        old_hash = record["file_hash"] if record else None

        deltas[fp] = {
            "old_hash": old_hash,
            "new_hash": new_hash,
            "changed":  old_hash != new_hash,
            "is_new":   old_hash is None,
        }

    return deltas


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL DIFF (uses ast_parser.py if available)
# ─────────────────────────────────────────────────────────────────────────────

def compute_structural_diffs(
    changed_files: List[Dict],
    repo_root:     str,
    store:         SQLiteEmbeddingStore,
    repo:          str,
) -> Dict[str, Dict]:
    """
    For Python files, compute structural AST diffs (added/removed functions,
    complexity changes). Falls back to hash-only diff for other languages.
    """
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from ast_parser import ASTParser, ASTDiff
        ast_parser = ASTParser(repo_root=repo_root)
        use_ast    = True
    except ImportError:
        use_ast    = False
        ast_parser = None

    results: Dict[str, Dict] = {}

    for f in changed_files:
        fp   = f["filepath"]
        lang = f.get("language", "unknown")

        if lang != "python" or not use_ast:
            results[fp] = {
                "is_meaningful": True,
                "change_summary": f"Non-Python file changed ({lang}): structural diff not available",
                "added_functions":    [],
                "removed_functions":  [],
                "changed_complexity": [],
            }
            continue

        abs_path = Path(repo_root) / fp
        if not abs_path.exists():
            results[fp] = {
                "is_meaningful": True,
                "change_summary": "File deleted or not on disk",
                "added_functions":    [],
                "removed_functions":  [],
                "changed_complexity": [],
            }
            continue

        record = store.get(repo, fp)
        stored_ast_dict = record.get("ast_features") if record else None

        # Reconstruct a FileAST-like object from stored JSON if available
        stored_file_ast = None
        if stored_ast_dict and use_ast:
            try:
                from ast_parser import FileAST, FunctionNode
                stored_file_ast = FileAST(
                    file_path    = fp,
                    language     = "python",
                    file_hash    = record.get("file_hash", ""),
                    imports      = stored_ast_dict.get("imports", []),
                    classes      = stored_ast_dict.get("classes", []),
                    functions    = [
                        FunctionNode(
                            name       = fn.get("name", ""),
                            class_name = fn.get("class_name"),
                            file_path  = fp,
                            start_line = fn.get("start_line", 0),
                            end_line   = fn.get("end_line", 0),
                            complexity = fn.get("complexity", 1),
                            num_args   = fn.get("num_args", 0),
                        )
                        for fn in stored_ast_dict.get("functions", [])
                    ],
                    methods      = [],
                )
            except Exception:
                stored_file_ast = None

        try:
            diff_result = ast_parser.compare_with_stored(str(abs_path), stored_file_ast)
            results[fp] = diff_result
        except Exception as e:
            log.warning("Structural diff failed for %s: %s", fp, e)
            results[fp] = {
                "is_meaningful": True,
                "change_summary": f"Structural diff error: {e}",
                "added_functions":    [],
                "removed_functions":  [],
                "changed_complexity": [],
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PRDiffProcessor:
    """
    Orchestrates PR-level change detection:
      1. Parse diff → extract changed files
      2. Re-embed changed files
      3. Compare against stored embeddings (cosine similarity)
      4. Compute hash deltas
      5. Compute structural diffs (Python)
      6. Identify test files and compute source→test similarities
    """

    def __init__(
        self,
        repo:      str,
        repo_root: str = ".",
        db_path:   str = "./greenops_output/module_registry.sqlite",
    ):
        self.repo      = repo
        self.repo_root = str(Path(repo_root).resolve())
        self.store     = get_store(db_path)
        self.extractor = RepoModuleExtractor(
            repo_root = self.repo_root,
            db_path   = db_path,
        )
        self.sim_engine = EmbeddingSimilarityEngine()

    def process_diff(
        self,
        diff_text:  str,
        pr_number:  int = 0,
        test_files: Optional[List[str]] = None,
    ) -> Dict:
        """
        Full PR processing pipeline.

        Args:
            diff_text:  unified diff string from github_ci_integration.py
            pr_number:  PR number for audit logging
            test_files: list of known test file paths (repo-relative)
                        If None, auto-discovers test files from the store

        Returns comprehensive dict with all change signals.
        """
        log.info("=== PR Diff Processor (PR #%d) ===", pr_number)

        # Step 1: Parse diff
        changed_files = parse_changed_files_from_diff(diff_text)
        if not changed_files:
            log.info("No parseable source files in diff")
            return self._empty_result(pr_number)

        changed_paths = [f["filepath"] for f in changed_files]

        # Step 2: Re-embed changed files
        pr_embeddings = self.extractor.embed_changed_files(
            repo          = self.repo,
            changed_files = changed_paths,
            pr_number     = pr_number,
        )

        # Step 3: Similarity vs stored embeddings
        similarity_scores = self.sim_engine.compare_pr_to_stored(
            store        = self.store,
            repo         = self.repo,
            pr_embeddings = pr_embeddings,
        )

        # Step 4: Hash deltas
        hash_deltas = compute_hash_deltas(
            store         = self.store,
            repo          = self.repo,
            changed_files = changed_files,
            repo_root     = self.repo_root,
        )

        # Step 5: Structural diffs
        structural_diffs = compute_structural_diffs(
            changed_files = changed_files,
            repo_root     = self.repo_root,
            store         = self.store,
            repo          = self.repo,
        )

        # Step 6: Test file discovery (auto-detect if not provided)
        if test_files is None:
            test_files = self._discover_test_files()

        # Step 7: Source → test similarity
        source_test_similarities = self.sim_engine.compare_changed_to_test_modules(
            store         = self.store,
            repo          = self.repo,
            pr_embeddings = pr_embeddings,
            test_files    = test_files,
        )

        # Flatten similarity scores: key = (source_file, test_file), value = sim
        # Build the format expected by GenerativeDependencyMapper
        func_test_similarities = {}
        for (src_file, test_file), sim in source_test_similarities.items():
            # Use filename stem as function/module identifier
            src_key  = Path(src_file).stem
            test_key = Path(test_file).stem
            func_test_similarities[(src_key, test_key)] = sim

        # Step 8: Compute composite impact score per changed module
        impact_scores = self._compute_impact_scores(
            similarity_scores = similarity_scores,
            hash_deltas       = hash_deltas,
            structural_diffs  = structural_diffs,
            changed_files     = changed_files,
        )

        # Step 9: Log PR run to store
        impacted_tests = list({
            test_file
            for (_, test_file) in source_test_similarities
        })
        self.store.log_pr_run(
            repo           = self.repo,
            pr_number      = pr_number,
            changed_files  = changed_paths,
            selected_tests = impacted_tests,
            pruned_tests   = [],
        )

        result = {
            "pr_number":              pr_number,
            "changed_modules":        changed_paths,
            "similarity_scores":      similarity_scores,
            "hash_deltas":            hash_deltas,
            "structural_diffs":       structural_diffs,
            "impact_scores":          impact_scores,
            "source_test_similarities": {
                f"{k[0]}→{k[1]}": v
                for k, v in source_test_similarities.items()
            },
            "func_test_similarities": {
                str(k): v for k, v in func_test_similarities.items()
            },
            "impacted_tests":         impacted_tests,
            "total_lines_changed":    sum(
                f["net_lines_changed"] for f in changed_files
            ),
            "num_meaningful_changes": sum(
                1 for d in structural_diffs.values()
                if d.get("is_meaningful", True)
            ),
        }

        self._print_diff_summary(result)
        return result

    def _discover_test_files(self) -> List[str]:
        """Auto-discover test files from the store (files with is_test_file=True)."""
        all_modules = self.store.list_all(self.repo)
        test_files = []
        for m in all_modules:
            fp = m["file_path"]
            if (
                "test" in fp.lower() or
                "spec" in fp.lower() or
                fp.startswith("tests/") or
                fp.startswith("test/")
            ):
                test_files.append(fp)
        log.info("Auto-discovered %d test files", len(test_files))
        return test_files

    @staticmethod
    def _compute_impact_scores(
        similarity_scores: Dict[str, float],
        hash_deltas:       Dict[str, Dict],
        structural_diffs:  Dict[str, Dict],
        changed_files:     List[Dict],
    ) -> Dict[str, float]:
        """
        Composite impact score per changed module.
        High impact = needs more test coverage.

        Formula:
          hash_signal        = 1.0 if hash changed, 0.0 if not
          embedding_signal   = 1.0 - cosine_similarity (lower sim = bigger change)
          structural_signal  = 1.0 if meaningful structural change, 0.3 otherwise
          line_signal        = min(net_lines_changed / 200, 1.0)

          impact = 0.3*hash_signal + 0.3*embedding_signal + 0.2*structural_signal + 0.2*line_signal
        """
        scores: Dict[str, float] = {}
        lines_map = {f["filepath"]: f["net_lines_changed"] for f in changed_files}

        for fp in similarity_scores:
            sim_val       = similarity_scores.get(fp, 0.5)
            hash_delta    = hash_deltas.get(fp, {})
            struct_diff   = structural_diffs.get(fp, {})
            net_lines     = lines_map.get(fp, 0)

            hash_signal        = 1.0 if hash_delta.get("changed", True) else 0.0
            embedding_signal   = 1.0 - sim_val   # low sim = high impact
            structural_signal  = 1.0 if struct_diff.get("is_meaningful", True) else 0.3
            line_signal        = min(net_lines / 200.0, 1.0)

            score = (
                0.30 * hash_signal +
                0.30 * embedding_signal +
                0.20 * structural_signal +
                0.20 * line_signal
            )
            scores[fp] = round(score, 4)

        return scores

    @staticmethod
    def _empty_result(pr_number: int) -> Dict:
        return {
            "pr_number":              pr_number,
            "changed_modules":        [],
            "similarity_scores":      {},
            "hash_deltas":            {},
            "structural_diffs":       {},
            "impact_scores":          {},
            "source_test_similarities": {},
            "func_test_similarities": {},
            "impacted_tests":         [],
            "total_lines_changed":    0,
            "num_meaningful_changes": 0,
        }

    @staticmethod
    def _print_diff_summary(result: dict):
        print(f"\n{'='*60}")
        print(f"PR Diff Summary — PR #{result['pr_number']}")
        print(f"{'='*60}")
        print(f"  Changed modules      : {len(result['changed_modules'])}")
        print(f"  Total lines changed  : {result['total_lines_changed']}")
        print(f"  Meaningful changes   : {result['num_meaningful_changes']}")
        print(f"  Impacted test files  : {len(result['impacted_tests'])}")
        print()
        print("  Module similarity vs stored embedding:")
        for fp, sim in sorted(result["similarity_scores"].items(),
                               key=lambda x: x[1]):
            delta  = result["hash_deltas"].get(fp, {})
            impact = result["impact_scores"].get(fp, "?")
            changed_str = "CHANGED" if delta.get("changed") else "unchanged"
            print(f"    {Path(fp).name:<35} sim={sim:.4f}  "
                  f"{changed_str}  impact={impact:.3f}")
        if result["impacted_tests"]:
            print()
            print("  Top impacted tests:")
            for t in result["impacted_tests"][:10]:
                print(f"    • {t}")
        print(f"{'='*60}\n")
