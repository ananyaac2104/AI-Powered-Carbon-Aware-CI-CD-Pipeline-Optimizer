"""
repo_module_extractor.py
========================
Green-Ops CI/CD Framework — Full Repository Module Extraction + Embedding

Traverses the ENTIRE repository, identifies all Python/JS/Java modules,
generates CodeBERT embeddings (open-source, no paid API), stores embeddings
and SHA-256 hashes in the SQLite store.

This replaces any demo/mock logic with real embedding generation.

Features:
  - Traverses all .py, .js, .ts, .java files (configurable)
  - Uses microsoft/codebert-base (CI-efficient, ~400MB, CPU-capable)
  - Falls back to TF-IDF + SVD if transformers unavailable
  - Skips files whose hash hasn't changed (incremental update)
  - Stores: embedding, hash, AST features, value_score
  - Outputs: JSON report of all modules + embedding stats

USAGE:
    python repo_module_extractor.py --repo /path/to/repo --output ./greenops_output

In pipeline:
    from repo_module_extractor import RepoModuleExtractor
    extractor = RepoModuleExtractor(repo_root="/path/to/repo")
    extractor.run_full_extraction(repo="org/repo")
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

from module_embedding_store import (
    SQLiteEmbeddingStore, get_store,
    compute_file_hash, hash_changed,
)

log = logging.getLogger("greenops.repo_extractor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rb", ".rs"}
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", "venv", ".venv", "env",
    ".tox", "dist", "build", ".eggs", "*.egg-info", ".mypy_cache",
    ".pytest_cache", "coverage", ".coverage", "htmlcov",
}
CODEBERT_MODEL = os.environ.get("GREENOPS_EMBED_MODEL", "microsoft/codebert-base")
BATCH_SIZE     = int(os.environ.get("GREENOPS_EMBED_BATCH_SIZE", "8"))
MAX_TOKEN_LEN  = 512
EMBEDDING_DIM  = 768


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING BACKENDS
# ─────────────────────────────────────────────────────────────────────────────

class CodeBERTEmbedder:
    """
    Production embedding backend using microsoft/codebert-base.
    Open-source, no API key needed. First run downloads ~400MB.
    """

    def __init__(self, model_name: str = CODEBERT_MODEL, device: str = "auto"):
        self.model_name = model_name
        self.model      = None
        self.tokenizer  = None

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        log.info("CodeBERTEmbedder: model=%s, device=%s", model_name, self.device)

    def load(self):
        """Download and load CodeBERT. Cached after first download."""
        try:
            from transformers import AutoTokenizer, AutoModel
            cache_dir = os.environ.get("TRANSFORMERS_CACHE", "./model_cache")
            log.info("Loading %s (cache=%s) ...", self.model_name, cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, cache_dir=cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            log.info("Model loaded ✓")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for CodeBERT embeddings.\n"
                "Install: pip install transformers torch"
            )

    def embed(self, code: str, language: str = "python") -> np.ndarray:
        """Embed a single code snippet → float32 numpy array [768]."""
        import torch

        if self.model is None:
            self.load()

        text = f"# language: {language}\n{code[:3000]}"  # truncate early
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKEN_LEN,
                padding="max_length",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            out    = self.model(**tokens)
            mask   = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return pooled.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_batch(
        self,
        codes:      List[str],
        languages:  List[str],
    ) -> List[np.ndarray]:
        """Embed a list of code snippets efficiently."""
        import torch

        if self.model is None:
            self.load()

        results = []
        for i in range(0, len(codes), BATCH_SIZE):
            batch_codes = codes[i:i + BATCH_SIZE]
            batch_langs = languages[i:i + BATCH_SIZE]
            texts = [
                f"# language: {lang}\n{code[:3000]}"
                for code, lang in zip(batch_codes, batch_langs)
            ]
            with torch.no_grad():
                tokens = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TOKEN_LEN,
                    padding=True,
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                out    = self.model(**tokens)
                mask   = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                embs   = pooled.cpu().numpy().astype(np.float32)
            results.extend(list(embs))
            log.info("  Embedded %d/%d", min(i + BATCH_SIZE, len(codes)), len(codes))

        return results


class TFIDFEmbedder:
    """
    Lightweight fallback embedder using TF-IDF + truncated SVD.
    No model downloads. Produces 256-dim vectors.
    Used when transformers/torch are unavailable.
    """

    DIM = 256

    def __init__(self):
        self.vectorizer = None
        self.svd        = None
        self._fitted    = False

    def fit(self, texts: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        tfidf = self.vectorizer.fit_transform(texts)
        actual_components = min(self.DIM, tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True
        log.info("TF-IDF embedder fitted on %d texts → dim=%d", len(texts), actual_components)

    def embed(self, code: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() with a corpus before embed()")
        tfidf = self.vectorizer.transform([code])
        vec   = self.svd.transform(tfidf)[0].astype(np.float32)
        # Pad to EMBEDDING_DIM with zeros for API compatibility
        padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded

    def embed_batch(
        self,
        codes:     List[str],
        languages: List[str],  # ignored but kept for API parity
    ) -> List[np.ndarray]:
        if not self._fitted:
            self.fit(codes)
        return [self.embed(c) for c in codes]


def get_embedder() -> CodeBERTEmbedder | TFIDFEmbedder:
    """Return CodeBERT if available, otherwise TF-IDF."""
    try:
        import transformers  # noqa: F401
        import torch         # noqa: F401
        return CodeBERTEmbedder()
    except ImportError:
        log.warning("transformers/torch not available — using TF-IDF embedder")
        return TFIDFEmbedder()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE FILE COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

def collect_repo_files(
    repo_root:  str,
    extensions: Optional[set] = None,
    max_files:  int = 5000,
) -> List[Dict]:
    """
    Traverse the repository and collect all source files.
    Returns a list of dicts: {file_path, language, size_bytes}
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    repo = Path(repo_root)
    if not repo.exists():
        raise FileNotFoundError(f"Repo root not found: {repo_root}")

    ext_to_lang = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rb": "ruby", ".rs": "rust",
    }

    collected = []
    skipped   = 0

    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue

        # Skip hidden and build directories
        parts = set(path.parts)
        if any(d in parts for d in SKIP_DIRS):
            skipped += 1
            continue
        if any(part.startswith(".") for part in path.parts[len(repo.parts):]):
            skipped += 1
            continue

        ext = path.suffix.lower()
        if ext not in extensions:
            continue

        rel_path = str(path.relative_to(repo))
        collected.append({
            "file_path":  rel_path,
            "abs_path":   str(path),
            "language":   ext_to_lang.get(ext, "unknown"),
            "size_bytes": path.stat().st_size,
        })

        if len(collected) >= max_files:
            log.warning("Max file limit %d reached — truncating", max_files)
            break

    log.info("Collected %d source files (skipped %d) in %s",
             len(collected), skipped, repo_root)
    return collected


# ─────────────────────────────────────────────────────────────────────────────
# AST FEATURE EXTRACTOR (lightweight, no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def extract_ast_features_lite(file_path: str, language: str) -> dict:
    """
    Lightweight structural feature extraction without full AST parsing.
    Used when ast_parser.py is not available.
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    if language == "python":
        imports   = re.findall(r"^(?:import|from)\s+([\w.]+)", content, re.MULTILINE)
        functions = re.findall(r"^def\s+(\w+)\s*\(", content, re.MULTILINE)
        classes   = re.findall(r"^class\s+(\w+)\s*[:(]", content, re.MULTILINE)
        test_fns  = [f for f in functions if f.startswith("test_")]
    elif language in ("javascript", "typescript"):
        imports   = re.findall(r"(?:import|require)\s*[({'\"](.+?)[\"')}]", content)
        functions = re.findall(r"(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s+)?(?:function|\()|\()", content)
        classes   = re.findall(r"class\s+(\w+)", content)
        test_fns  = [f for f in functions if re.search(r"test|spec|it|describe", f, re.I)]
    elif language == "java":
        imports   = re.findall(r"^import\s+([\w.]+);", content, re.MULTILINE)
        functions = re.findall(r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\(", content)
        classes   = re.findall(r"class\s+(\w+)", content)
        test_fns  = [f for f in functions if f.startswith("test")]
    else:
        imports = functions = classes = test_fns = []

    lines = content.count("\n") + 1
    return {
        "imports":      imports[:50],
        "functions":    functions[:100],
        "classes":      classes[:30],
        "test_functions": test_fns[:50],
        "num_lines":    lines,
        "num_imports":  len(imports),
        "num_functions": len(functions),
        "num_classes":  len(classes),
        "is_test_file": (
            "test" in Path(file_path).name.lower() or
            "spec" in Path(file_path).name.lower() or
            bool(test_fns)
        ),
    }


def compute_value_score(ast_features: dict) -> float:
    """
    Compute module value score (higher = more structurally complex = higher change risk).
    """
    import math
    fn_count     = ast_features.get("num_functions", 0)
    import_count = ast_features.get("num_imports", 0)
    line_count   = ast_features.get("num_lines", 1)
    class_count  = ast_features.get("num_classes", 0)
    return round(
        0.4 * fn_count +
        0.3 * import_count +
        0.2 * math.log(line_count + 1) +
        0.1 * class_count,
        4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RepoModuleExtractor:
    """
    Traverses a repository, generates embeddings for all modules, and
    persists them to the embedding store.

    Incremental: only re-embeds files whose hash has changed.
    """

    def __init__(
        self,
        repo_root:  str = ".",
        db_path:    str = "./greenops_output/module_registry.sqlite",
        embedder    = None,
    ):
        self.repo_root = str(Path(repo_root).resolve())
        self.store     = get_store(db_path)
        self.embedder  = embedder or get_embedder()
        self._ast_parser = None  # lazy import

    def _get_ast_parser(self):
        """Try to load the full AST parser; fall back to lightweight version."""
        if self._ast_parser is not None:
            return self._ast_parser
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from ast_parser import ASTParser
            self._ast_parser = ASTParser(repo_root=self.repo_root)
            log.info("Using full ASTParser from ast_parser.py")
        except ImportError:
            log.info("ast_parser.py not available — using lightweight extractor")
            self._ast_parser = None
        return self._ast_parser

    def run_full_extraction(
        self,
        repo:         str,
        force_reembed: bool = False,
        pr_number:    int  = 0,
    ) -> Dict:
        """
        Full extraction pipeline:
          1. Collect all source files in the repo
          2. For each file, compute SHA-256 hash
          3. Skip files whose hash is unchanged (unless force_reembed)
          4. Generate CodeBERT embedding for changed/new files
          5. Store embedding + hash + AST features
          6. Return extraction report

        Args:
            repo:          "org/repo" identifier
            force_reembed: Re-embed all files even if hash unchanged
            pr_number:     PR that triggered this extraction

        Returns:
            {total, new, updated, skipped, errors, modules: [...]}
        """
        log.info("=== Full Repo Module Extraction ===")
        log.info("Repo root: %s", self.repo_root)

        # Pull latest from S3 if available
        if hasattr(self.store, "pull_from_s3"):
            self.store.pull_from_s3(repo)

        # Step 1: Collect files
        files = collect_repo_files(self.repo_root)
        log.info("Found %d source files to process", len(files))

        # Step 2: Determine which files need (re-)embedding
        to_embed      = []
        existing_hashes = {
            r["file_path"]: r["file_hash"]
            for r in self.store.list_all(repo)
        }

        for f in files:
            current_hash = compute_file_hash(f["abs_path"])
            f["file_hash"] = current_hash
            stored_hash    = existing_hashes.get(f["file_path"])

            if force_reembed or hash_changed(stored_hash, current_hash):
                f["reason"] = "new" if stored_hash is None else "changed"
                to_embed.append(f)
            else:
                f["reason"] = "unchanged"

        log.info(
            "Files to embed: %d / %d (force=%s)",
            len(to_embed), len(files), force_reembed,
        )

        # Step 3: Extract AST features for files to embed
        ast_parser = self._get_ast_parser()
        for f in to_embed:
            if ast_parser:
                try:
                    file_ast = ast_parser.parse_file(f["abs_path"])
                    f["ast_features"] = file_ast.to_dict()
                    f["value_score"]  = file_ast.value_score()
                    f["content"]      = Path(f["abs_path"]).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except Exception as e:
                    log.warning("AST parse failed for %s: %s", f["file_path"], e)
                    f["ast_features"] = extract_ast_features_lite(f["abs_path"], f["language"])
                    f["value_score"]  = compute_value_score(f["ast_features"])
                    f["content"]      = Path(f["abs_path"]).read_text(
                        encoding="utf-8", errors="replace"
                    ) if Path(f["abs_path"]).exists() else ""
            else:
                f["ast_features"] = extract_ast_features_lite(f["abs_path"], f["language"])
                f["value_score"]  = compute_value_score(f["ast_features"])
                f["content"]      = Path(f["abs_path"]).read_text(
                    encoding="utf-8", errors="replace"
                ) if Path(f["abs_path"]).exists() else ""

        # Step 4: Generate embeddings in batch
        new_count = updated_count = error_count = 0

        if to_embed:
            codes     = [f.get("content", "") for f in to_embed]
            languages = [f["language"] for f in to_embed]

            # Load the embedder if it hasn't been loaded yet
            if isinstance(self.embedder, CodeBERTEmbedder) and self.embedder.model is None:
                self.embedder.load()

            # For TFIDFEmbedder: fit on corpus first
            if isinstance(self.embedder, TFIDFEmbedder) and not self.embedder._fitted:
                all_codes = [
                    Path(f["abs_path"]).read_text(encoding="utf-8", errors="replace")
                    for f in files
                    if Path(f["abs_path"]).exists()
                ]
                self.embedder.fit(all_codes or codes)

            log.info("Generating embeddings for %d files ...", len(to_embed))
            t0         = time.time()

            try:
                embeddings = self.embedder.embed_batch(codes, languages)
            except Exception as e:
                log.error("Batch embedding failed: %s — using zeros", e)
                embeddings = [np.zeros(EMBEDDING_DIM, dtype=np.float32)] * len(to_embed)

            elapsed = time.time() - t0
            log.info("Embedding complete: %.1fs for %d files (%.2fs/file)",
                     elapsed, len(to_embed), elapsed / max(len(to_embed), 1))

            # Step 5: Store embeddings
            for f, emb in zip(to_embed, embeddings):
                try:
                    self.store.upsert(
                        repo         = repo,
                        file_path    = f["file_path"],
                        file_hash    = f["file_hash"],
                        embedding    = emb,
                        language     = f["language"],
                        ast_features = f.get("ast_features"),
                        value_score  = f.get("value_score", 0.0),
                        pr_number    = pr_number,
                    )
                    if f["reason"] == "new":
                        new_count += 1
                    else:
                        updated_count += 1
                except Exception as e:
                    log.error("Store failed for %s: %s", f["file_path"], e)
                    error_count += 1

        skipped_count = len(files) - len(to_embed)

        # Sync to S3 after extraction
        if hasattr(self.store, "sync_to_s3"):
            self.store.sync_to_s3(repo)

        report = {
            "total":   len(files),
            "new":     new_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "errors":  error_count,
            "modules": [
                {
                    "file_path":   f["file_path"],
                    "language":    f["language"],
                    "file_hash":   f.get("file_hash", ""),
                    "value_score": f.get("value_score", 0.0),
                    "reason":      f.get("reason", ""),
                }
                for f in files
            ],
        }

        self._print_extraction_report(report)
        return report

    def embed_changed_files(
        self,
        repo:          str,
        changed_files: List[str],  # repo-relative paths
        pr_number:     int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Re-embed ONLY the files that changed in a PR.
        This is the PR-triggered path (fast, incremental).

        Returns: {file_path: embedding}
        """
        log.info("Re-embedding %d changed files for PR #%d", len(changed_files), pr_number)

        if isinstance(self.embedder, CodeBERTEmbedder) and self.embedder.model is None:
            self.embedder.load()

        result: Dict[str, np.ndarray] = {}
        ast_parser = self._get_ast_parser()

        codes     = []
        languages = []
        valid_files = []

        for rel_path in changed_files:
            abs_path = Path(self.repo_root) / rel_path
            if not abs_path.exists():
                log.warning("Changed file not found: %s", rel_path)
                continue

            ext      = abs_path.suffix.lower()
            language = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".java": "java", ".go": "go",
            }.get(ext, "unknown")

            if language == "unknown":
                continue

            content = abs_path.read_text(encoding="utf-8", errors="replace")
            codes.append(content)
            languages.append(language)
            valid_files.append((rel_path, abs_path, language, content))

        if not codes:
            return result

        # For TFIDFEmbedder: may need to fit on stored corpus first
        if isinstance(self.embedder, TFIDFEmbedder) and not self.embedder._fitted:
            all_files = collect_repo_files(self.repo_root)
            corpus = [
                Path(f["abs_path"]).read_text(encoding="utf-8", errors="replace")
                for f in all_files[:500]
                if Path(f["abs_path"]).exists()
            ]
            self.embedder.fit(corpus or codes)

        embeddings = self.embedder.embed_batch(codes, languages)

        for (rel_path, abs_path, language, content), emb in zip(valid_files, embeddings):
            file_hash   = compute_file_hash(str(abs_path))
            ast_features = (
                ast_parser.parse_file(str(abs_path)).to_dict()
                if ast_parser else
                extract_ast_features_lite(str(abs_path), language)
            )
            value_score = compute_value_score(ast_features) if not ast_parser else (
                ast_parser.parse_file(str(abs_path)).value_score()
                if ast_parser else 0.0
            )

            self.store.upsert(
                repo         = repo,
                file_path    = rel_path,
                file_hash    = file_hash,
                embedding    = emb,
                language     = language,
                ast_features = ast_features,
                value_score  = value_score,
                pr_number    = pr_number,
            )
            result[rel_path] = emb
            log.info("Re-embedded: %s", rel_path)

        return result

    @staticmethod
    def _print_extraction_report(report: dict):
        print(f"\n{'='*60}")
        print("Module Extraction Complete")
        print(f"{'='*60}")
        print(f"  Total files scanned : {report['total']}")
        print(f"  New embeddings      : {report['new']}")
        print(f"  Updated embeddings  : {report['updated']}")
        print(f"  Skipped (unchanged) : {report['skipped']}")
        print(f"  Errors              : {report['errors']}")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Green-Ops Repo Module Extractor")
    parser.add_argument("--repo-root",    default=".",
                        help="Path to repository root")
    parser.add_argument("--repo-id",      default="local/repo",
                        help="Repository identifier (e.g. org/repo)")
    parser.add_argument("--output",       default="./greenops_output",
                        help="Output directory for SQLite store")
    parser.add_argument("--force",        action="store_true",
                        help="Force re-embedding of all files")
    parser.add_argument("--pr-number",    type=int, default=0)
    args = parser.parse_args()

    db_path   = str(Path(args.output) / "module_registry.sqlite")
    extractor = RepoModuleExtractor(repo_root=args.repo_root, db_path=db_path)
    report    = extractor.run_full_extraction(
        repo          = args.repo_id,
        force_reembed = args.force,
        pr_number     = args.pr_number,
    )

    out_path = Path(args.output) / "extraction_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nExtraction report saved → {out_path}")

    # Show store stats
    store = get_store(db_path)
    stats = store.stats(args.repo_id)
    print(f"\nStore stats: {stats}")
