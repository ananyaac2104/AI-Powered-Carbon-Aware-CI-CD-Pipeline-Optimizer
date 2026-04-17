"""
module_embedding_store.py
=========================
Green-Ops CI/CD Framework — Module Embedding + Hash Persistence Layer

Stores and retrieves per-module embeddings and SHA-256 hashes across CI runs.
Supports two backends:
  1. Local SQLite (default, zero-config, CI-efficient)
  2. AWS S3 + DynamoDB (production, when boto3 + credentials available)

Tables / Schema:
  module_registry (
      repo          TEXT,
      file_path     TEXT,
      file_hash     TEXT,        -- SHA-256 of file content
      embedding     BLOB,        -- numpy float32 array, pickled
      embedding_dim INT,
      language      TEXT,
      last_seen_pr  INT,
      updated_at    TEXT,
      PRIMARY KEY (repo, file_path)
  )

USAGE:
    from module_embedding_store import ModuleEmbeddingStore
    store = ModuleEmbeddingStore(db_path="./greenops_output/module_registry.sqlite")

    # Save
    store.upsert(repo="org/repo", file_path="src/auth.py",
                 file_hash="abc123...", embedding=np.array([...]))

    # Load
    record = store.get(repo="org/repo", file_path="src/auth.py")
    if record and record["file_hash"] == current_hash:
        embedding = record["embedding"]  # skip re-embedding

    # List all stored files
    all_records = store.list_all(repo="org/repo")

    # Batch similarity search (returns top-k similar modules)
    results = store.search_similar(query_embedding, repo="org/repo", top_k=10)
"""

import hashlib
import json
import logging
import os
import pickle
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("greenops.embedding_store")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH  = os.environ.get("GREENOPS_DB_PATH",
                                   "./greenops_output/module_registry.sqlite")
S3_BUCKET        = os.environ.get("GREENOPS_S3_BUCKET", "")
S3_PREFIX        = os.environ.get("GREENOPS_S3_PREFIX", "greenops/embeddings/")
DYNAMO_TABLE     = os.environ.get("GREENOPS_DYNAMO_TABLE", "greenops_module_registry")

EMBEDDING_DIM    = 768  # GraphCodeBERT / microsoft/unixcoder-base


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL SQLITE BACKEND
# ─────────────────────────────────────────────────────────────────────────────

class SQLiteEmbeddingStore:
    """
    Thread-safe SQLite store for module embeddings and SHA-256 hashes.
    Designed for single-node CI runners. File locking ensures safety across
    concurrent pipeline steps on the same machine.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS module_registry (
        repo            TEXT NOT NULL,
        file_path       TEXT NOT NULL,
        file_hash       TEXT NOT NULL,
        embedding       BLOB,
        embedding_dim   INTEGER DEFAULT 768,
        language        TEXT DEFAULT 'python',
        ast_features    TEXT,          -- JSON: functions, imports, classes counts
        value_score     REAL DEFAULT 0.0,
        last_seen_pr    INTEGER DEFAULT 0,
        updated_at      TEXT,
        PRIMARY KEY (repo, file_path)
    );
    CREATE TABLE IF NOT EXISTS pr_run_log (
        run_id          TEXT PRIMARY KEY,
        repo            TEXT,
        pr_number       INTEGER,
        changed_files   TEXT,          -- JSON list
        selected_tests  TEXT,          -- JSON list
        pruned_tests    TEXT,          -- JSON list
        carbon_gco2     REAL,
        pruning_rate    REAL,
        created_at      TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_repo_hash ON module_registry(repo, file_hash);
    CREATE INDEX IF NOT EXISTS idx_repo_updated ON module_registry(repo, updated_at);
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent read-write
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)
        log.info("SQLite store initialised at %s", self.db_path)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def upsert(
        self,
        repo:          str,
        file_path:     str,
        file_hash:     str,
        embedding:     Optional[np.ndarray] = None,
        language:      str = "python",
        ast_features:  Optional[dict] = None,
        value_score:   float = 0.0,
        pr_number:     int = 0,
    ) -> None:
        """Insert or update a module record."""
        emb_blob = pickle.dumps(embedding.astype(np.float32)) if embedding is not None else None
        ast_json = json.dumps(ast_features) if ast_features else None

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO module_registry
                    (repo, file_path, file_hash, embedding, embedding_dim,
                     language, ast_features, value_score, last_seen_pr, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo, file_path) DO UPDATE SET
                    file_hash     = excluded.file_hash,
                    embedding     = excluded.embedding,
                    embedding_dim = excluded.embedding_dim,
                    language      = excluded.language,
                    ast_features  = excluded.ast_features,
                    value_score   = excluded.value_score,
                    last_seen_pr  = excluded.last_seen_pr,
                    updated_at    = excluded.updated_at
            """, (
                repo, file_path, file_hash, emb_blob,
                embedding.shape[0] if embedding is not None else EMBEDDING_DIM,
                language, ast_json, value_score, pr_number,
                datetime.utcnow().isoformat(),
            ))
        log.debug("Upserted module %s (hash=%s)", file_path, file_hash[:12])

    def get(self, repo: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single module record."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM module_registry WHERE repo=? AND file_path=?",
                (repo, file_path),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def get_by_hash(self, repo: str, file_hash: str) -> List[Dict[str, Any]]:
        """Find all modules with a specific hash (useful for copy-detection)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM module_registry WHERE repo=? AND file_hash=?",
                (repo, file_hash),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_all(self, repo: str) -> List[Dict[str, Any]]:
        """Return all stored modules for a repo (without embeddings for efficiency)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT repo, file_path, file_hash, language,
                          value_score, last_seen_pr, updated_at
                   FROM module_registry WHERE repo=? ORDER BY value_score DESC""",
                (repo,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, repo: str, file_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM module_registry WHERE repo=? AND file_path=?",
                (repo, file_path),
            )

    def get_embeddings_matrix(self, repo: str) -> tuple:
        """
        Load all embeddings for a repo as a numpy matrix.
        Returns: (file_paths: list[str], matrix: np.ndarray[N, 768])
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_path, embedding FROM module_registry "
                "WHERE repo=? AND embedding IS NOT NULL",
                (repo,),
            ).fetchall()

        if not rows:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        file_paths = []
        embeddings = []
        for row in rows:
            try:
                emb = pickle.loads(row["embedding"])
                file_paths.append(row["file_path"])
                embeddings.append(emb.astype(np.float32))
            except Exception as e:
                log.warning("Failed to deserialise embedding for %s: %s",
                            row["file_path"], e)

        matrix = np.vstack(embeddings) if embeddings else np.empty((0, EMBEDDING_DIM))
        return file_paths, matrix

    def search_similar(
        self,
        query_embedding: np.ndarray,
        repo:            str,
        top_k:           int = 20,
        min_similarity:  float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-k most similar modules to a query embedding.
        Uses cosine similarity over all stored embeddings.

        Returns list of dicts: {file_path, similarity, file_hash, language}
        """
        file_paths, matrix = self.get_embeddings_matrix(repo)
        if matrix.shape[0] == 0:
            log.warning("No embeddings stored for repo %s", repo)
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        sims  = cosine_similarity(query, matrix).flatten()

        results = []
        for i, (fp, sim) in enumerate(zip(file_paths, sims)):
            if sim >= min_similarity:
                results.append({"file_path": fp, "similarity": float(sim)})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # ── PR Run Logging ────────────────────────────────────────────────────────

    def log_pr_run(
        self,
        repo:           str,
        pr_number:      int,
        changed_files:  List[str],
        selected_tests: List[str],
        pruned_tests:   List[str],
        carbon_gco2:    float = 0.0,
        pruning_rate:   float = 0.0,
    ) -> str:
        """Persist a PR run record for audit trail."""
        run_id = f"{repo}_{pr_number}_{int(time.time())}"
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pr_run_log
                    (run_id, repo, pr_number, changed_files, selected_tests,
                     pruned_tests, carbon_gco2, pruning_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, repo, pr_number,
                json.dumps(changed_files),
                json.dumps(selected_tests),
                json.dumps(pruned_tests),
                carbon_gco2, pruning_rate,
                datetime.utcnow().isoformat(),
            ))
        return run_id

    def get_pr_history(self, repo: str, limit: int = 50) -> List[dict]:
        """Return recent PR run history for training data collection."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pr_run_log WHERE repo=? ORDER BY created_at DESC LIMIT ?",
                (repo, limit),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["changed_files"]  = json.loads(d["changed_files"]  or "[]")
            d["selected_tests"] = json.loads(d["selected_tests"] or "[]")
            d["pruned_tests"]   = json.loads(d["pruned_tests"]   or "[]")
            result.append(d)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d.get("embedding"):
            try:
                d["embedding"] = pickle.loads(d["embedding"])
            except Exception:
                d["embedding"] = None
        if d.get("ast_features"):
            try:
                d["ast_features"] = json.loads(d["ast_features"])
            except Exception:
                pass
        return d

    def stats(self, repo: str) -> dict:
        """Return store statistics for a repo."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM module_registry WHERE repo=?", (repo,)
            ).fetchone()[0]
            with_emb = conn.execute(
                "SELECT COUNT(*) FROM module_registry WHERE repo=? AND embedding IS NOT NULL",
                (repo,)
            ).fetchone()[0]
            runs = conn.execute(
                "SELECT COUNT(*) FROM pr_run_log WHERE repo=?", (repo,)
            ).fetchone()[0]
        return {
            "total_modules":        total,
            "modules_with_embedding": with_emb,
            "pr_runs_logged":       runs,
            "db_path":              str(self.db_path),
        }


# ─────────────────────────────────────────────────────────────────────────────
# S3 BACKEND (optional, production)
# ─────────────────────────────────────────────────────────────────────────────

class S3EmbeddingStore(SQLiteEmbeddingStore):
    """
    Extends SQLiteEmbeddingStore with S3 sync for multi-node CI environments.

    Write path: SQLite (fast, local) → async S3 sync after each PR
    Read path:  SQLite (L1) → S3 pull on cache miss

    Requires: pip install boto3
    Environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or IAM role)
    """

    def __init__(
        self,
        db_path:   str = DEFAULT_DB_PATH,
        s3_bucket: str = S3_BUCKET,
        s3_prefix: str = S3_PREFIX,
    ):
        super().__init__(db_path)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self._s3_client = None

        if s3_bucket:
            self._init_s3()

    def _init_s3(self):
        try:
            import boto3
            self._s3_client = boto3.client("s3")
            log.info("S3 store initialised (bucket=%s, prefix=%s)",
                     self.s3_bucket, self.s3_prefix)
        except ImportError:
            log.warning("boto3 not installed — S3 sync disabled")

    def sync_to_s3(self, repo: str) -> bool:
        """Upload the local SQLite db to S3 for persistence across runners."""
        if not self._s3_client:
            return False
        s3_key = f"{self.s3_prefix}{repo.replace('/', '_')}_registry.sqlite"
        try:
            self._s3_client.upload_file(
                str(self.db_path), self.s3_bucket, s3_key
            )
            log.info("DB synced to s3://%s/%s", self.s3_bucket, s3_key)
            return True
        except Exception as e:
            log.warning("S3 sync failed: %s", e)
            return False

    def pull_from_s3(self, repo: str) -> bool:
        """Download the S3 db to local, restoring state from previous runs."""
        if not self._s3_client:
            return False
        s3_key = f"{self.s3_prefix}{repo.replace('/', '_')}_registry.sqlite"
        try:
            self._s3_client.download_file(
                self.s3_bucket, s3_key, str(self.db_path)
            )
            log.info("DB pulled from s3://%s/%s", self.s3_bucket, s3_key)
            self._init_db()  # ensure schema is current
            return True
        except Exception as e:
            log.info("S3 pull miss (first run?): %s", e)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY — select backend based on environment
# ─────────────────────────────────────────────────────────────────────────────

def get_store(db_path: str = DEFAULT_DB_PATH) -> SQLiteEmbeddingStore:
    """
    Return the appropriate store backend based on environment.
    Defaults to local SQLite for CI-efficiency.
    """
    if S3_BUCKET:
        return S3EmbeddingStore(db_path=db_path)
    return SQLiteEmbeddingStore(db_path=db_path)


# ─────────────────────────────────────────────────────────────────────────────
# HASH UTILITIES (standalone — no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's content."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception as e:
        log.warning("Cannot hash %s: %s", file_path, e)
        return ""
    return h.hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_changed(stored_hash: Optional[str], current_hash: str) -> bool:
    """Return True if the hash has changed since last stored."""
    if stored_hash is None:
        return True  # new file
    return stored_hash != current_hash
