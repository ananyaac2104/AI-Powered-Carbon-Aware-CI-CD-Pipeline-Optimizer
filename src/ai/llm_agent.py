"""
llm_impact_analyzer.py
=======================
Green-Ops Framework — LLM-Powered Cross-Module Impact Analyzer

PURPOSE
-------
This module is the "brain" that sits between:
  - graphcodebert_embeddings.py  (semantic vectors for every module)
  - xgboost_gatekeeper.py        (Pf predictor that consumes impact features)

It answers ONE question per PR:
  "Given the set of changed modules, which OTHER modules in the codebase
   are impacted — and WHY?"

The answer drives the XGBoost feature columns:
  module_impact_score, is_kafka_consumer/producer, is_shared_db,
  is_frontend_contract, is_shared_utility, transitive_depth

IMPACT DETECTION LAYERS (applied in order, all results merged)
--------------------------------------------------------------
  1. EMBEDDING SIMILARITY   — cosine(changed_emb, all_module_emb) > threshold
                              "semantically similar code is likely affected"
  2. SHARED DATABASE TABLES — module A and B both reference the same table names
                              extracted from AST (ORM annotations, raw SQL strings)
  3. DIRECT IMPORTS         — module B imports module A (static dep graph)
  4. KAFKA / MESSAGING      — module A produces on topic T; module B consumes T
  5. FRONTEND API CONTRACT  — module A exposes endpoint E; module B calls E

For each impacted module the LLM generates:
  - impact_type       : one of {embedding, shared_db, import, kafka_consumer,
                                kafka_producer, frontend_contract, shared_utility}
  - transitive_depth  : 1 = directly connected, 2 = one hop away, etc.
  - confidence        : 0.0–1.0  (used as module_impact_score in XGBoost)
  - explanation       : plain-English reason (stored in pruning_decision.json)

HISTORIC TELEMETRY PASSTHROUGH
-------------------------------
Pre-submit and post-submit CSVs from:
  https://www.kaggle.com/datasets/akshyaaa/xgboost-pruning-dataset-based-on-historic-failure

are read here and ALL rows are appended to the payload regardless of whether
the test is semantically relevant to the current PR.  The XGBoost gatekeeper
uses those rows for class-balance calibration and as a prior on flakiness.

EMBEDDING GENERATION ON NEW MODULE
-----------------------------------
When a NEW module arrives (no stored embedding for its hash), this module:
  1. Reads the module's source code from the preprocessing artifact
  2. Calls GraphCodeBERTEmbedder.embed_code() to generate a fresh vector
  3. Stores the vector in the embedding registry keyed by module_hash
  4. Proceeds with all five impact-detection layers as normal

So the answer to "is this generating embeddings after a new module comes in
and there is a git difference?" is YES — see _ensure_embeddings() below.

Dependencies:
    pip install anthropic numpy pandas scikit-learn
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("greenops.llm_impact")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_MODEL        = "claude-sonnet-4-20250514"
EMBEDDING_SIM_THRESHOLD = float(os.environ.get("EMBEDDING_SIM_THRESHOLD", "0.72"))
PRESUBMIT_CSV          = os.environ.get("PRESUBMIT_CSV",  "presubmit_clean.csv")
POSTSUBMIT_CSV         = os.environ.get("POSTSUBMIT_CSV", "postsubmit_clean.csv")
OUTPUT_DIR             = Path(os.environ.get("GREENOPS_OUTPUT", "./greenops_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column names in the Kaggle telemetry CSVs that identify a test uniquely
TELEMETRY_TEST_ID_COL  = "test_name"   # adjust if your CSV uses a different name
TELEMETRY_RESULT_COL   = "test_result" # 1 = PASSED, 0 = FAILED


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — EMBEDDING REGISTRY
# Stores and retrieves GraphCodeBERT vectors keyed by module_hash.
# Auto-generates embeddings for any new (unseen) module that arrives in a PR.
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingRegistry:
    """
    Persistent key-value store: module_hash → np.ndarray (768-dim vector).

    On first access for a hash that is not yet in the registry, the registry
    calls back into GraphCodeBERTEmbedder to generate and cache the vector.
    This is the mechanism that answers: "does the system generate embeddings
    after a new module comes in via a git diff?" — YES, right here.
    """

    def __init__(self, registry_path: str = "./greenops_output/embedding_registry.npy"):
        self.registry_path  = Path(registry_path)
        self._store: dict[str, np.ndarray] = {}
        self._embedder = None   # lazy-loaded to avoid importing torch at import time
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        meta_path = self.registry_path.with_suffix(".json")
        npy_path  = self.registry_path

        if meta_path.exists() and npy_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)          # [{hash, index}, ...]
                matrix = np.load(str(npy_path))  # shape (N, 768)
                self._store = {
                    entry["hash"]: matrix[entry["index"]]
                    for entry in meta
                }
                log.info("EmbeddingRegistry: loaded %d cached vectors", len(self._store))
            except Exception as exc:
                log.warning("Could not load embedding registry: %s", exc)

    def save(self):
        if not self._store:
            return
        hashes = list(self._store.keys())
        matrix = np.vstack([self._store[h] for h in hashes])
        meta   = [{"hash": h, "index": i} for i, h in enumerate(hashes)]

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.registry_path), matrix)
        with open(self.registry_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f)
        log.info("EmbeddingRegistry: saved %d vectors", len(hashes))

    # ── Core API ─────────────────────────────────────────────────────────────

    def get(self, module_hash: str) -> Optional[np.ndarray]:
        return self._store.get(module_hash)

    def put(self, module_hash: str, vector: np.ndarray):
        self._store[module_hash] = vector

    def all_hashes(self) -> list[str]:
        return list(self._store.keys())

    def all_vectors(self) -> np.ndarray:
        if not self._store:
            return np.empty((0, 768))
        return np.vstack(list(self._store.values()))

    def ensure_embeddings(
        self,
        modules: list[dict],
        source_lookup: dict[str, str],   # module_hash → source_code string
    ) -> dict[str, np.ndarray]:
        """
        For every module in `modules`, return its embedding.
        If the hash is not yet in the registry (NEW module from a fresh diff),
        generate the embedding using GraphCodeBERT and cache it.

        Args:
            modules:       list of module dicts (must have 'module_hash' and 'language')
            source_lookup: {module_hash: source_code} — the raw code from the diff

        Returns:
            {module_hash: np.ndarray} for ALL modules in the input list
        """
        missing = [
            m for m in modules
            if m.get("module_hash") and m["module_hash"] not in self._store
        ]

        if missing:
            log.info(
                "NEW modules detected in diff (%d): generating GraphCodeBERT embeddings ...",
                len(missing),
            )
            embedder = self._get_embedder()
            for mod in missing:
                h      = mod["module_hash"]
                lang   = mod.get("language", "python")
                source = source_lookup.get(h, "")

                if not source:
                    log.warning("No source code for hash %s — using zero vector", h)
                    self._store[h] = np.zeros(768, dtype=np.float32)
                    continue

                result = embedder.embed_code(source, identifier=h, language=lang)
                self._store[h] = result.embedding
                log.info("  Generated embedding for %s (%s)", h, mod.get("filepath", "?"))

            self.save()

        return {
            m["module_hash"]: self._store[m["module_hash"]]
            for m in modules
            if m.get("module_hash") and m["module_hash"] in self._store
        }

    def _get_embedder(self):
        """Lazy-load GraphCodeBERTEmbedder only when actually needed."""
        if self._embedder is None:
            try:
                from graphcodebert_embeddings import GraphCodeBERTEmbedder
                self._embedder = GraphCodeBERTEmbedder()
                self._embedder.load_model()
            except Exception as exc:
                log.warning("GraphCodeBERT unavailable (%s) — using random mock embeddings", exc)

                class _MockEmbedder:
                    def embed_code(self, code, identifier="", language="python"):
                        class R:
                            embedding = np.random.rand(768).astype(np.float32)
                        return R()

                self._embedder = _MockEmbedder()
        return self._embedder


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — STRUCTURAL IMPACT DETECTORS
# Five deterministic detectors that run BEFORE the LLM, so the LLM only has
# to reason about ambiguous cases and generate explanations.
# ─────────────────────────────────────────────────────────────────────────────

class StructuralImpactDetectors:
    """
    Pure-Python heuristics that detect cross-module relationships
    by inspecting AST-extracted metadata (no LLM calls needed here).
    """

    # ── 2a. Embedding Similarity ─────────────────────────────────────────────

    @staticmethod
    def by_embedding_similarity(
        changed_hashes:    list[str],
        all_module_hashes: list[str],
        registry:          EmbeddingRegistry,
        threshold:         float = EMBEDDING_SIM_THRESHOLD,
    ) -> list[dict]:
        """
        Find registry modules whose embedding is cosine-similar to ANY
        changed module above `threshold`.

        Returns list of impact dicts: {module_hash, impact_type, score, depth}
        """
        results = []
        if not changed_hashes or not all_module_hashes:
            return results

        changed_vecs = np.vstack([
            registry.get(h) for h in changed_hashes
            if registry.get(h) is not None
        ])

        for target_hash in all_module_hashes:
            if target_hash in changed_hashes:
                continue
            target_vec = registry.get(target_hash)
            if target_vec is None:
                continue

            # Max similarity across all changed modules
            sims = cosine_similarity(
                changed_vecs,
                target_vec.reshape(1, -1)
            ).flatten()
            max_sim = float(sims.max())

            if max_sim >= threshold:
                results.append({
                    "module_hash":  target_hash,
                    "impact_type":  "embedding",
                    "score":        round(max_sim, 4),
                    "depth":        1,
                    "signal":       f"cosine_similarity={max_sim:.4f}",
                })

        log.info("Embedding similarity: %d impacted modules (threshold=%.2f)",
                 len(results), threshold)
        return results

    # ── 2b. Shared Database Tables ───────────────────────────────────────────

    @staticmethod
    def by_shared_db_tables(
        changed_modules: list[dict],
        all_modules:     list[dict],
    ) -> list[dict]:
        """
        Module B is impacted if it references ANY db_table that module A (changed) also uses.
        db_tables is a list of strings extracted by ast_parser.py from:
          - JPA @Table annotations, @Entity names
          - SQLAlchemy model class names
          - raw SQL strings (table name heuristic)
        """
        changed_tables: set[str] = set()
        for m in changed_modules:
            changed_tables.update(t.lower() for t in m.get("db_tables", []))

        if not changed_tables:
            return []

        changed_hashes = {m["module_hash"] for m in changed_modules}
        results = []

        for mod in all_modules:
            if mod.get("module_hash") in changed_hashes:
                continue
            mod_tables = {t.lower() for t in mod.get("db_tables", [])}
            shared     = changed_tables & mod_tables
            if shared:
                results.append({
                    "module_hash":  mod["module_hash"],
                    "impact_type":  "shared_db",
                    "score":        min(1.0, 0.6 + 0.1 * len(shared)),
                    "depth":        1,
                    "signal":       f"shared_tables={sorted(shared)}",
                })

        log.info("Shared DB tables: %d impacted modules (tables=%s)",
                 len(results), sorted(changed_tables))
        return results

    # ── 2c. Direct Imports ───────────────────────────────────────────────────

    @staticmethod
    def by_direct_imports(
        changed_modules: list[dict],
        all_modules:     list[dict],
    ) -> list[dict]:
        """
        Module B is impacted if its import list contains ANY filepath
        (or class/package name) from a changed module.

        We match on the stem of the filepath (e.g., "PaymentService") because
        import strings rarely contain the full path.
        """
        # Build set of identifiers for changed modules
        changed_ids: set[str] = set()
        for m in changed_modules:
            fp = Path(m.get("filepath", ""))
            changed_ids.add(fp.stem.lower())
            changed_ids.add(fp.name.lower())
            # Also index by dot-separated package name (Java/Python)
            pkg = fp.with_suffix("").as_posix().replace("/", ".").lstrip(".")
            changed_ids.add(pkg.lower())

        changed_hashes = {m["module_hash"] for m in changed_modules}
        results = []

        for mod in all_modules:
            if mod.get("module_hash") in changed_hashes:
                continue
            imports = [imp.lower() for imp in mod.get("imports", [])]
            matched = [imp for imp in imports
                       if any(cid in imp for cid in changed_ids)]
            if matched:
                results.append({
                    "module_hash":  mod["module_hash"],
                    "impact_type":  "import",
                    "score":        min(1.0, 0.55 + 0.15 * len(matched)),
                    "depth":        1,
                    "signal":       f"imports={matched[:5]}",
                })

        log.info("Direct imports: %d impacted modules", len(results))
        return results

    # ── 2d. Kafka / Messaging ────────────────────────────────────────────────

    @staticmethod
    def by_kafka_messaging(
        changed_modules: list[dict],
        all_modules:     list[dict],
    ) -> list[dict]:
        """
        Finds producer/consumer pairs on shared Kafka topics.

        A changed module that PRODUCEs on topic T impacts all modules that
        CONSUME topic T (and vice versa — a changed consumer is impacted
        by anything producing on the same topic).

        module.kafka_topics  : list of topic names
        module.kafka_role    : "producer" | "consumer" | "both" | ""
        """
        results = []
        changed_hashes = {m["module_hash"] for m in changed_modules}

        # Index: topic → list of {module_hash, role}
        topic_index: dict[str, list[dict]] = {}
        for mod in changed_modules + all_modules:
            for topic in mod.get("kafka_topics", []):
                topic_index.setdefault(topic, []).append({
                    "module_hash": mod["module_hash"],
                    "role":        mod.get("kafka_role", ""),
                })

        for mod_changed in changed_modules:
            for topic in mod_changed.get("kafka_topics", []):
                role = mod_changed.get("kafka_role", "")
                for peer in topic_index.get(topic, []):
                    if peer["module_hash"] in changed_hashes:
                        continue
                    peer_role = peer["role"]

                    # Producer changed → consumers impacted
                    if role == "producer" and peer_role in ("consumer", "both"):
                        impact_type = "kafka_consumer"
                    # Consumer changed → producers may need awareness
                    elif role == "consumer" and peer_role in ("producer", "both"):
                        impact_type = "kafka_producer"
                    else:
                        continue

                    results.append({
                        "module_hash":  peer["module_hash"],
                        "impact_type":  impact_type,
                        "score":        0.85,
                        "depth":        1,
                        "signal":       f"kafka_topic={topic}, changed_role={role}",
                    })

        # Deduplicate by hash (keep highest score)
        seen: dict[str, dict] = {}
        for r in results:
            h = r["module_hash"]
            if h not in seen or r["score"] > seen[h]["score"]:
                seen[h] = r
        results = list(seen.values())

        log.info("Kafka messaging: %d impacted modules", len(results))
        return results

    # ── 2e. Frontend API Contracts ───────────────────────────────────────────

    @staticmethod
    def by_frontend_api_contracts(
        changed_modules: list[dict],
        all_modules:     list[dict],
    ) -> list[dict]:
        """
        Changed backend module exposes an API endpoint also used by a frontend module.

        module.api_endpoints : list of strings like "POST /api/v1/payments"
        module.is_frontend   : bool
        """
        changed_endpoints: set[str] = set()
        for m in changed_modules:
            for ep in m.get("api_endpoints", []):
                changed_endpoints.add(ep.lower().strip())

        if not changed_endpoints:
            return []

        changed_hashes = {m["module_hash"] for m in changed_modules}
        results = []

        for mod in all_modules:
            if mod.get("module_hash") in changed_hashes:
                continue
            if not mod.get("is_frontend", False):
                continue
            mod_endpoints = {ep.lower().strip() for ep in mod.get("api_endpoints", [])}
            shared = changed_endpoints & mod_endpoints
            if shared:
                results.append({
                    "module_hash":  mod["module_hash"],
                    "impact_type":  "frontend_contract",
                    "score":        0.90,
                    "depth":        1,
                    "signal":       f"api_contracts={sorted(shared)}",
                })

        log.info("Frontend API contracts: %d impacted modules", len(results))
        return results

    # ── Merge all detector results ───────────────────────────────────────────

    @staticmethod
    def merge_impact_results(
        layers: list[list[dict]],
    ) -> dict[str, dict]:
        """
        Merge results from all five detectors.
        For each module hash, keep the highest-score entry; accumulate
        all signals and deduplicate impact_types.

        Returns: {module_hash: merged_impact_dict}
        """
        merged: dict[str, dict] = {}

        for layer_results in layers:
            for item in layer_results:
                h = item["module_hash"]
                if h not in merged:
                    merged[h] = {
                        "module_hash":   h,
                        "impact_types":  [],
                        "score":         0.0,
                        "depth":         item["depth"],
                        "signals":       [],
                    }
                merged[h]["impact_types"].append(item["impact_type"])
                merged[h]["signals"].append(item["signal"])
                merged[h]["score"] = max(merged[h]["score"], item["score"])
                merged[h]["depth"] = min(merged[h]["depth"], item["depth"])

        # Deduplicate lists
        for h, v in merged.items():
            v["impact_types"] = list(dict.fromkeys(v["impact_types"]))

        return merged


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TRANSITIVE DEPTH WALKER
# Extends the direct-impact graph to N hops using the import graph
# ─────────────────────────────────────────────────────────────────────────────

def compute_transitive_depths(
    direct_impacts:  dict[str, dict],    # {hash: impact_dict} from merge_impact_results
    all_modules:     list[dict],
    max_depth:       int = 3,
) -> dict[str, dict]:
    """
    BFS from each directly impacted module to find modules impacted
    transitively (depth 2, 3, ...) via the import graph.

    Only `import` relationships are walked transitively — Kafka / DB impacts
    are already direct and do not propagate further by default.
    """
    # Build import adjacency: hash → list of hashes that import it
    hash_by_stem: dict[str, str] = {}
    for mod in all_modules:
        fp = Path(mod.get("filepath", ""))
        for key in [fp.stem.lower(), fp.name.lower()]:
            hash_by_stem[key] = mod.get("module_hash", "")

    reverse_imports: dict[str, list[str]] = {}  # hash → [hashes that import it]
    for mod in all_modules:
        for imp in mod.get("imports", []):
            imp_l = imp.lower()
            for key, h in hash_by_stem.items():
                if key in imp_l and h:
                    reverse_imports.setdefault(h, []).append(
                        mod.get("module_hash", "")
                    )

    # BFS
    extended = dict(direct_impacts)
    frontier = list(direct_impacts.keys())

    for depth in range(2, max_depth + 1):
        next_frontier = []
        for h in frontier:
            for dependent_hash in reverse_imports.get(h, []):
                if dependent_hash and dependent_hash not in extended:
                    extended[dependent_hash] = {
                        "module_hash":  dependent_hash,
                        "impact_types": ["transitive_import"],
                        "score":        max(0.3, extended[h]["score"] * 0.6),
                        "depth":        depth,
                        "signals":      [f"transitive_via={h} depth={depth}"],
                    }
                    next_frontier.append(dependent_hash)
        frontier = next_frontier
        if not frontier:
            break

    log.info(
        "Transitive depth walk: %d direct → %d total impacted modules (max_depth=%d)",
        len(direct_impacts), len(extended), max_depth,
    )
    return extended


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LLM EXPLANATION GENERATOR
# Calls Claude claude-sonnet-4-20250514 to:
#   a) verify the structural findings
#   b) assign final confidence scores
#   c) produce a human-readable explanation per impacted module
#   d) write a one-paragraph PR summary
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior software architect reviewing a pull request as part of an automated CI/CD \
test-pruning system called Green-Ops. Your job is to reason about cross-module impact and \
return ONLY a JSON object — no prose, no markdown, no preamble.

The JSON schema you must return:
{
  "summary": "<one paragraph: what changed, why it matters, blast radius>",
  "impact_analysis": [
    {
      "module_hash":        "<hash string>",
      "filepath":           "<file path>",
      "impact_type":        "<primary type from: embedding|shared_db|import|kafka_consumer|kafka_producer|frontend_contract|shared_utility|transitive_import>",
      "confidence":         <float 0.0–1.0>,
      "transitive_depth":   <int 1–5>,
      "explanation":        "<one sentence: why this module's tests should run>",
      "should_run_tests":   <true|false>
    }
  ],
  "affected_module_hashes":      ["<hash>", ...],
  "kafka_topics_affected":       ["<topic>", ...],
  "shared_db_tables_affected":   ["<table>", ...],
  "frontend_contracts_affected": ["<endpoint>", ...]
}

Rules:
- Only include modules where should_run_tests is true in affected_module_hashes.
- Set confidence = 0.0 for modules you believe are NOT actually impacted despite structural signals.
- Be conservative: prefer false negatives (running tests unnecessarily) over false positives \
  (missing a broken test).
- Do not invent module hashes or filepaths — use only what is provided.
"""


def build_llm_prompt(
    changed_modules:   list[dict],
    candidate_impacts: dict[str, dict],   # from merge + transitive
    all_modules_index: dict[str, dict],   # hash → module dict
) -> str:
    """Build the user message sent to the LLM."""
    changed_summary = [
        {
            "module_hash":   m.get("module_hash"),
            "filepath":      m.get("filepath"),
            "language":      m.get("language"),
            "db_tables":     m.get("db_tables", []),
            "kafka_topics":  m.get("kafka_topics", []),
            "kafka_role":    m.get("kafka_role", ""),
            "api_endpoints": m.get("api_endpoints", []),
            "is_frontend":   m.get("is_frontend", False),
            "imports":       m.get("imports", [])[:10],   # cap for token budget
        }
        for m in changed_modules
    ]

    candidate_summary = []
    for h, impact in candidate_impacts.items():
        mod = all_modules_index.get(h, {})
        candidate_summary.append({
            "module_hash":    h,
            "filepath":       mod.get("filepath", "unknown"),
            "impact_types":   impact["impact_types"],
            "structural_score": round(impact["score"], 3),
            "transitive_depth": impact["depth"],
            "signals":        impact["signals"],
            "db_tables":      mod.get("db_tables", []),
            "kafka_topics":   mod.get("kafka_topics", []),
            "kafka_role":     mod.get("kafka_role", ""),
            "api_endpoints":  mod.get("api_endpoints", []),
            "is_frontend":    mod.get("is_frontend", False),
        })

    return json.dumps({
        "instruction": (
            "Analyze the impact of the changed modules on the candidate modules. "
            "Return ONLY the JSON schema described in the system prompt."
        ),
        "changed_modules":    changed_summary,
        "candidate_impacts":  candidate_summary,
    }, indent=2)


def call_llm(prompt: str) -> dict:
    """
    Call Claude claude-sonnet-4-20250514 via the Anthropic SDK.
    Falls back to a deterministic mock if the SDK is unavailable or errors.
    """
    try:
        import anthropic
        client   = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from env
        response = client.messages.create(
            model      = ANTHROPIC_MODEL,
            max_tokens = 4096,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        # Strip accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    except ImportError:
        log.warning("anthropic SDK not installed — using structural-only mock response")
        return _mock_llm_response(prompt)
    except Exception as exc:
        log.error("LLM call failed: %s — using structural-only mock response", exc)
        return _mock_llm_response(prompt)


def _mock_llm_response(prompt: str) -> dict:
    """
    Deterministic fallback when the LLM is unavailable.
    Uses the structural scores directly as confidence values.
    """
    try:
        data = json.loads(prompt)
        candidates = data.get("candidate_impacts", [])
    except Exception:
        candidates = []

    impact_analysis = []
    affected_hashes = []

    for c in candidates:
        score = c.get("structural_score", 0.5)
        should_run = score >= 0.5
        impact_type = (c.get("impact_types") or ["embedding"])[0]
        if should_run:
            affected_hashes.append(c["module_hash"])
        impact_analysis.append({
            "module_hash":      c["module_hash"],
            "filepath":         c.get("filepath", "unknown"),
            "impact_type":      impact_type,
            "confidence":       round(score, 3),
            "transitive_depth": c.get("transitive_depth", 1),
            "explanation":      (
                f"Structural signal ({impact_type}, score={score:.2f}) "
                f"suggests this module may be affected by the PR changes."
            ),
            "should_run_tests": should_run,
        })

    return {
        "summary": "Structural-only analysis (LLM unavailable). Impact based on AST signals.",
        "impact_analysis":             impact_analysis,
        "affected_module_hashes":      affected_hashes,
        "kafka_topics_affected":       [],
        "shared_db_tables_affected":   [],
        "frontend_contracts_affected": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — HISTORIC TELEMETRY LOADER
# Reads presubmit_clean.csv + postsubmit_clean.csv and attaches ALL rows
# to the final payload so XGBoost has its full training prior, regardless
# of whether each test row is semantically related to the current PR.
# ─────────────────────────────────────────────────────────────────────────────

def load_historic_telemetry(
    presubmit_path:  str = PRESUBMIT_CSV,
    postsubmit_path: str = POSTSUBMIT_CSV,
) -> dict:
    """
    Load both telemetry CSVs and return a dict with:
      - "presubmit_df"  : full presubmit DataFrame  (may be empty)
      - "postsubmit_df" : full postsubmit DataFrame (may be empty)
      - "summary"       : stats dict for the payload

    ALL rows are returned — the caller (xgboost_gatekeeper) decides which
    rows are used for training vs inference. Passing everything here ensures
    the Pf prior is calibrated on the full failure history, not just the
    tests that happen to be relevant to this PR.
    """
    frames = {}
    summary = {}

    for label, path in [("presubmit", presubmit_path), ("postsubmit", postsubmit_path)]:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(path, low_memory=False)
            frames[f"{label}_df"] = df
            n_fail = int((df[TELEMETRY_RESULT_COL] == 0).sum()) \
                     if TELEMETRY_RESULT_COL in df.columns else -1
            summary[label] = {
                "rows":           len(df),
                "failure_rows":   n_fail,
                "failure_rate":   round(n_fail / max(len(df), 1), 4) if n_fail >= 0 else None,
                "unique_tests":   df[TELEMETRY_TEST_ID_COL].nunique()
                                  if TELEMETRY_TEST_ID_COL in df.columns else None,
            }
            log.info(
                "Telemetry [%s]: %d rows, %d failures (%.1f%%)",
                label, len(df), n_fail, 100 * n_fail / max(len(df), 1),
            )
        else:
            log.warning("Telemetry CSV not found: %s (download from Kaggle)", path)
            frames[f"{label}_df"] = pd.DataFrame()
            summary[label] = {"rows": 0, "failure_rows": 0, "failure_rate": None, "unique_tests": 0}

    frames["summary"] = summary
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN ORCHESTRATOR: LLMImpactAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class LLMImpactAnalyzer:
    """
    Full cross-module impact analysis pipeline.

    Called by xgboost_gatekeeper.py → run_gatekeeper_pipeline():

        analyzer = LLMImpactAnalyzer()
        impact_result = analyzer.analyze_impact(changed_modules, module_registry)

    Pipeline:
      1. Ensure embeddings exist for all changed + registry modules
         (generates new ones from source code if this is a new module in a diff)
      2. Run five structural detectors in parallel-ish
      3. Walk transitive import graph to find depth-2/3 impacts
      4. Call Claude claude-sonnet-4-20250514 to verify + explain + assign confidence
      5. Attach full historic telemetry as a passthrough payload
      6. Return unified payload ready for XGBoost feature engineering
    """

    def __init__(
        self,
        registry:          Optional[EmbeddingRegistry] = None,
        detectors:         Optional[StructuralImpactDetectors] = None,
        sim_threshold:     float = EMBEDDING_SIM_THRESHOLD,
        max_transitive:    int   = 3,
    ):
        self.registry       = registry or EmbeddingRegistry()
        self.detectors      = detectors or StructuralImpactDetectors()
        self.sim_threshold  = sim_threshold
        self.max_transitive = max_transitive

    def decide(self, similarity, carbon_intensity):
        """
        Carbon-aware AI decision refinement.
        Legacy API expected by decision_engine.py
        Returns 'RUN_TEST' or 'SKIP_TEST'.
        """
        # If the code similarity is significant (>0.7), we should probably run tests
        # regardless of carbon intensity to avoid regression risk.
        if (float(similarity) if similarity is not None else 0.0) > 0.7:
            return "RUN_TEST"

        # If carbon intensity is high (> 450 gCO2/kWh) and similarity is moderate/low,
        # we skip the test to save energy.
        if (float(carbon_intensity) if carbon_intensity is not None else 0.0) > 450 and \
           (float(similarity) if similarity is not None else 0.0) < 0.5:
            return "SKIP_TEST"

        # Default to running tests for safety
        return "RUN_TEST"


    def analyze_impact(
        self,
        changed_modules:  list[dict],
        module_registry:  list[dict],
        source_lookup:    Optional[dict[str, str]] = None,
    ) -> dict:
        """
        Main entry point.

        Args:
            changed_modules:  Modules changed in this PR (from module_db.py).
                              Each dict must have: module_hash, filepath, language,
                              imports, db_tables, kafka_topics, kafka_role,
                              api_endpoints, is_frontend.
            module_registry:  ALL known modules in the codebase (same schema).
            source_lookup:    {module_hash: source_code_string}.
                              Used to generate embeddings for NEW modules.
                              If None, source is read from 'filepath' directly.

        Returns:
            Full impact payload dict (see schema in _SYSTEM_PROMPT).
            Also includes "telemetry" key with presubmit/postsubmit DataFrames.
        """
        log.info("=" * 60)
        log.info("LLMImpactAnalyzer.analyze_impact() — %d changed modules",
                 len(changed_modules))
        log.info("=" * 60)

        # ── Step 1: Build source lookup if not provided ───────────────────────
        if source_lookup is None:
            source_lookup = self._build_source_lookup(changed_modules + module_registry)

        # ── Step 2: Ensure embeddings exist (generates for NEW modules) ───────
        log.info("Step 2: Ensuring embeddings for all modules ...")
        all_modules_with_emb = changed_modules + [
            m for m in module_registry
            if m.get("module_hash") not in
               {c["module_hash"] for c in changed_modules}
        ]
        all_embeddings = self.registry.ensure_embeddings(
            all_modules_with_emb, source_lookup
        )
        log.info("  Embeddings ready for %d modules", len(all_embeddings))

        # ── Step 3: Run five structural detectors ────────────────────────────
        log.info("Step 3: Running structural impact detectors ...")
        changed_hashes  = [m["module_hash"] for m in changed_modules if m.get("module_hash")]
        all_hashes      = [m["module_hash"] for m in module_registry  if m.get("module_hash")]

        layer_embedding  = self.detectors.by_embedding_similarity(
            changed_hashes, all_hashes, self.registry, self.sim_threshold
        )
        layer_db         = self.detectors.by_shared_db_tables(changed_modules, module_registry)
        layer_imports    = self.detectors.by_direct_imports(changed_modules, module_registry)
        layer_kafka      = self.detectors.by_kafka_messaging(changed_modules, module_registry)
        layer_frontend   = self.detectors.by_frontend_api_contracts(changed_modules, module_registry)

        merged_direct = self.detectors.merge_impact_results([
            layer_embedding, layer_db, layer_imports, layer_kafka, layer_frontend
        ])
        log.info("  Direct impact: %d candidate modules", len(merged_direct))

        # ── Step 4: Transitive depth walk ─────────────────────────────────────
        log.info("Step 4: Walking transitive import graph (max_depth=%d) ...",
                 self.max_transitive)
        all_impacts = compute_transitive_depths(
            merged_direct, module_registry, max_depth=self.max_transitive
        )
        log.info("  Total candidates after transitive walk: %d", len(all_impacts))

        # ── Step 5: LLM verification + explanation ────────────────────────────
        log.info("Step 5: Calling LLM for impact verification and explanations ...")
        all_modules_index = {m["module_hash"]: m for m in module_registry}
        prompt     = build_llm_prompt(changed_modules, all_impacts, all_modules_index)
        llm_result = call_llm(prompt)

        # ── Step 6: Attach historic telemetry (passthrough, all rows) ─────────
        log.info("Step 6: Loading historic telemetry ...")
        telemetry = load_historic_telemetry(PRESUBMIT_CSV, POSTSUBMIT_CSV)

        # ── Step 7: Enrich result with structural signals + telemetry summary ─
        final = {
            **llm_result,
            "structural_signals": {
                h: {
                    "impact_types":     v["impact_types"],
                    "structural_score": round(v["score"], 4),
                    "depth":            v["depth"],
                    "signals":          v["signals"],
                }
                for h, v in all_impacts.items()
            },
            "telemetry_summary":  telemetry["summary"],
            # DataFrames are attached here so xgboost_gatekeeper.py can use them
            # directly without re-reading CSVs.
            "_presubmit_df":  telemetry["presubmit_df"],
            "_postsubmit_df": telemetry["postsubmit_df"],
        }

        # ── Save artifact ─────────────────────────────────────────────────────
        artifact_path = OUTPUT_DIR / "impact_analysis.json"
        serializable  = {
            k: v for k, v in final.items()
            if not k.startswith("_")   # skip DataFrames for JSON serialization
        }
        with open(artifact_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        log.info("Impact analysis saved → %s", artifact_path)

        # ── Log summary ───────────────────────────────────────────────────────
        n_affected = len(final.get("affected_module_hashes", []))
        n_total    = len(all_impacts)
        log.info(
            "Impact summary: %d/%d candidates confirmed by LLM | "
            "Kafka: %s | DB: %s | Frontend: %s",
            n_affected,
            n_total,
            final.get("kafka_topics_affected", []),
            final.get("shared_db_tables_affected", []),
            final.get("frontend_contracts_affected", []),
        )
        log.info("LLM summary: %s", final.get("summary", "")[:200])

        return final

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_source_lookup(modules: list[dict]) -> dict[str, str]:
        """
        Build {module_hash: source_code} by reading files from disk.
        Used when the caller does not pass a pre-built source_lookup.
        Falls back to empty string if file is not found (e.g. deleted file).
        """
        lookup = {}
        for mod in modules:
            h  = mod.get("module_hash")
            fp = mod.get("filepath", "")
            if not h:
                continue
            if h in lookup:
                continue
            try:
                lookup[h] = Path(fp).read_text(encoding="utf-8", errors="replace")
            except Exception:
                lookup[h] = ""
        return lookup


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Running LLMImpactAnalyzer smoke test ...")

    # Mirrors the mock data from xgboost_gatekeeper.py __main__ block
    changed = [
        {
            "module_hash":   "abc123",
            "filepath":      "src/payments/PaymentService.java",
            "language":      "java",
            "imports":       ["com.example.db.PaymentRepo", "com.example.kafka.PaymentProducer"],
            "db_tables":     ["payments", "transactions"],
            "kafka_topics":  ["payment-events"],
            "kafka_role":    "producer",
            "api_endpoints": ["POST /api/v1/payments"],
            "is_frontend":   False,
        }
    ]

    registry_mods = changed + [
        {
            "module_hash":   "def456",
            "filepath":      "src/notifications/NotificationConsumer.java",
            "language":      "java",
            "imports":       ["com.example.kafka.PaymentConsumer"],
            "db_tables":     [],
            "kafka_topics":  ["payment-events"],
            "kafka_role":    "consumer",
            "api_endpoints": [],
            "is_frontend":   False,
        },
        {
            "module_hash":   "ghi789",
            "filepath":      "frontend/src/components/PaymentForm.tsx",
            "language":      "typescript",
            "imports":       [],
            "db_tables":     [],
            "kafka_topics":  [],
            "kafka_role":    "",
            "api_endpoints": ["POST /api/v1/payments"],
            "is_frontend":   True,
        },
        {
            "module_hash":   "jkl012",
            "filepath":      "src/audit/AuditService.java",
            "language":      "java",
            "imports":       ["com.example.payments.PaymentService"],
            "db_tables":     ["payments", "audit_log"],
            "kafka_topics":  [],
            "kafka_role":    "",
            "api_endpoints": [],
            "is_frontend":   False,
        },
    ]

    # Provide mock source code so we can test embedding generation
    source_lookup = {
        "abc123": "public class PaymentService { void processPayment(Payment p) { repo.save(p); } }",
        "def456": "public class NotificationConsumer { @KafkaListener void onPayment(PaymentEvent e) {} }",
        "ghi789": "const PaymentForm = () => { fetch('POST /api/v1/payments', ...); }",
        "jkl012": "public class AuditService { void audit(Payment p) { auditRepo.log(p); } }",
    }

    analyzer = LLMImpactAnalyzer()
    result   = analyzer.analyze_impact(changed, registry_mods, source_lookup)

    print("\n" + "=" * 60)
    print("IMPACT ANALYSIS RESULT")
    print("=" * 60)
    print(f"  Summary          : {result.get('summary', '')[:120]} ...")
    print(f"  Affected hashes  : {result.get('affected_module_hashes', [])}")
    print(f"  Kafka topics     : {result.get('kafka_topics_affected', [])}")
    print(f"  DB tables        : {result.get('shared_db_tables_affected', [])}")
    print(f"  Frontend APIs    : {result.get('frontend_contracts_affected', [])}")
    print(f"  Telemetry summary: {result.get('telemetry_summary', {})}")
    print("\n  Per-module explanations:")
    for item in result.get("impact_analysis", []):
        print(f"    [{item['impact_type']:20s}] {item['filepath']:50s} "
              f"conf={item['confidence']:.2f}  depth={item['transitive_depth']}  "
              f"run={item['should_run_tests']}")
        print(f"      → {item['explanation']}")

LLMAgent = LLMImpactAnalyzer
