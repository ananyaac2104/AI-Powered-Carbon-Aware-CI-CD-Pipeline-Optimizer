"""
graphcodebert_embeddings.py
============================
Green-Ops CI/CD Framework — GraphCodeBERT Semantic Embedding Module

Converts source code snippets (functions, test bodies, diffs) into dense
semantic vector representations using Microsoft's GraphCodeBERT model.

These embeddings power:
  - Test ↔ source code similarity scoring (which tests are relevant to a PR?)
  - Dynamic Dependency Graph edge weights
  - XGBoost Pf feature inputs (semantic_similarity_score)

Install dependencies:
    pip install transformers torch pandas numpy scikit-learn

    # Optional: GPU support
    pip install torch --index-url https://download.pytorch.org/whl/cu118

Usage:
    from graphcodebert_embeddings import GraphCodeBERTEmbedder

    embedder = GraphCodeBERTEmbedder()
    embedder.load_model()

    # Embed a single function
    vec = embedder.embed_code("def add(a, b): return a + b")

    # Batch embed all test functions extracted from AST
    test_vecs = embedder.embed_batch(code_snippets)

    # Compare PR diff to test suite
    scores = embedder.compute_similarity(diff_embedding, test_embeddings)
"""

import json
import logging
import os

# Mac Silicon / Protobuf Crash Fallback
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("greenops.embeddings")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME      = "microsoft/graphcodebert-base"
EMBEDDING_DIM   = 768          # GraphCodeBERT hidden size
MAX_TOKEN_LEN   = 512          # Maximum token length (model constraint)
BATCH_SIZE      = 16           # Number of snippets per forward pass
SIMILARITY_THRESHOLD = 0.75    # Min cosine similarity to consider a test "relevant"


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Result of embedding a single code snippet."""
    identifier:     str            # test_name or file_path#function_name
    embedding:      np.ndarray     # shape: (768,)
    token_count:    int
    was_truncated:  bool
    language:       str = "python"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHCODEBERT EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────

class GraphCodeBERTEmbedder:
    """
    Wraps the GraphCodeBERT model to produce semantic embeddings for
    code snippets extracted from your source files and test suite.

    GraphCodeBERT is pre-trained on code+comments+data-flow graphs,
    making it superior to plain CodeBERT for understanding logical intent.
    """

    def __init__(
        self,
        model_name:   str  = MODEL_NAME,
        device:       str  = "auto",
        cache_dir:    str  = "./model_cache",
        use_fp16:     bool = False,
    ):
        self.model_name = model_name
        self.cache_dir  = cache_dir
        self.use_fp16   = use_fp16
        self.model      = None
        self.tokenizer  = None

        # Device selection
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        log.info("GraphCodeBERTEmbedder initialized (device=%s, model=%s)",
                 self.device, model_name)

    def load_model(self):
        """
        Download and load GraphCodeBERT from HuggingFace.
        First call downloads ~500MB. Subsequent calls use local cache.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required.\n"
                "Install: pip install transformers torch"
            ) from e

        log.info("Loading %s ...", self.model_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        if self.use_fp16 and self.device == "cuda":
            self.model = self.model.half()

        self.model = self.model.to(self.device)
        self.model.eval()   # Disable dropout for deterministic embeddings

        log.info("Model loaded ✓  (params: ~125M, dim: %d)", EMBEDDING_DIM)

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call embedder.load_model() first.")

    def embed_code(
        self,
        code_snippet: str,
        identifier:   str = "",
        language:     str = "python",
    ) -> EmbeddingResult:
        """
        Embed a single code snippet into a 768-dimensional vector.

        The embedding is the mean-pooled representation of all token
        hidden states from the final transformer layer. This captures
        the semantic intent of the entire function/class/diff.

        Args:
            code_snippet: Raw source code string (function, class, diff)
            identifier:   Human-readable label (e.g., "auth.py#login")
            language:     Source language hint (for tokenizer prefix)

        Returns:
            EmbeddingResult with a 768-dim numpy array
        """
        import torch

        self._ensure_loaded()

        if not code_snippet or not code_snippet.strip():
            log.warning("Empty code snippet for %s — returning zero vector", identifier)
            return EmbeddingResult(
                identifier=identifier,
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                token_count=0,
                was_truncated=False,
                language=language,
            )

        # GraphCodeBERT uses a <s> code </s> format
        # Prepend language tag to improve language disambiguation
        formatted = f"# language: {language}\n{code_snippet}"

        with torch.no_grad():
            tokens = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKEN_LEN,
                padding="max_length",
            )

            was_truncated = (
                self.tokenizer(formatted, return_tensors="pt")["input_ids"].shape[1]
                > MAX_TOKEN_LEN
            )
            token_count = int(tokens["attention_mask"].sum().item())

            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model(**tokens)

            # Mean-pooling over non-padding tokens
            hidden_states    = outputs.last_hidden_state   # (1, seq_len, 768)
            attention_mask   = tokens["attention_mask"]    # (1, seq_len)
            mask_expanded    = attention_mask.unsqueeze(-1).float()
            sum_hidden       = (hidden_states * mask_expanded).sum(dim=1)
            count            = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled      = (sum_hidden / count).squeeze(0)   # (768,)

            embedding = mean_pooled.cpu().numpy().astype(np.float32)

        return EmbeddingResult(
            identifier   = identifier or "unknown",
            embedding    = embedding,
            token_count  = token_count,
            was_truncated= was_truncated,
            language     = language,
        )

    def embed_batch(
        self,
        code_snippets: list,
        identifiers:   Optional[list] = None,
        language:      str = "python",
    ) -> list:
        """
        Embed a list of code snippets in batches (efficient GPU use).

        Args:
            code_snippets: List of source code strings
            identifiers:   Optional list of labels (same length as code_snippets)
            language:      Source language

        Returns:
            List of EmbeddingResult objects
        """
        import torch

        self._ensure_loaded()

        if identifiers is None:
            identifiers = [f"snippet_{i}" for i in range(len(code_snippets))]

        results = []
        total   = len(code_snippets)
        log.info("Embedding %d code snippets in batches of %d ...", total, BATCH_SIZE)

        for batch_start in range(0, total, BATCH_SIZE):
            batch_end      = min(batch_start + BATCH_SIZE, total)
            batch_snippets = code_snippets[batch_start:batch_end]
            batch_ids      = identifiers[batch_start:batch_end]

            # Format snippets
            formatted = [f"# language: {language}\n{s}" for s in batch_snippets]

            with torch.no_grad():
                tokens = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TOKEN_LEN,
                    padding=True,
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                outputs = self.model(**tokens)

                # Mean pooling
                hidden     = outputs.last_hidden_state
                mask       = tokens["attention_mask"].unsqueeze(-1).float()
                embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embeddings = embeddings.cpu().numpy().astype(np.float32)

            for i, (emb, idf, snip) in enumerate(
                zip(embeddings, batch_ids, batch_snippets)
            ):
                results.append(EmbeddingResult(
                    identifier   = idf,
                    embedding    = emb,
                    token_count  = int(tokens["attention_mask"][i].sum().item()),
                    was_truncated= False,
                    language     = language,
                ))

            log.info("  Embedded %d/%d", batch_end, total)

        return results

    def embed_diff(self, diff_text: str) -> EmbeddingResult:
        """
        Embed a git unified diff.

        Only the added (+) lines are embedded — these represent the
        actual code change and are what the model needs to assess.
        """
        # Extract only added lines from the diff
        added_lines = [
            line[1:]  # strip the leading '+'
            for line in diff_text.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]
        added_code = "\n".join(added_lines)

        if not added_code.strip():
            log.warning("No added lines found in diff")
            added_code = diff_text  # fall back to full diff

        return self.embed_code(added_code, identifier="pr_diff")

    def compute_similarity(
        self,
        query_embedding:  Union[np.ndarray, EmbeddingResult],
        corpus_embeddings: Union[np.ndarray, list],
        identifiers:      Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Compute cosine similarity between a query (e.g., PR diff embedding)
        and a corpus (e.g., all test function embeddings).

        Returns a DataFrame sorted by similarity descending:
            identifier | similarity_score | is_relevant
        """
        # Normalize inputs
        if isinstance(query_embedding, EmbeddingResult):
            query_vec = query_embedding.embedding.reshape(1, -1)
        else:
            query_vec = np.array(query_embedding).reshape(1, -1)

        if isinstance(corpus_embeddings, list):
            if all(isinstance(e, EmbeddingResult) for e in corpus_embeddings):
                ids         = [e.identifier for e in corpus_embeddings]
                corpus_vecs = np.vstack([e.embedding for e in corpus_embeddings])
            else:
                ids         = identifiers or [f"item_{i}" for i in range(len(corpus_embeddings))]
                corpus_vecs = np.vstack(corpus_embeddings)
        else:
            corpus_vecs = corpus_embeddings
            ids         = identifiers or [f"item_{i}" for i in range(len(corpus_vecs))]

        # Cosine similarity: (1, N) → flatten to (N,)
        similarities = cosine_similarity(query_vec, corpus_vecs).flatten()

        df = pd.DataFrame({
            "identifier":       ids,
            "similarity_score": similarities,
        })
        df["is_relevant"] = (df["similarity_score"] >= SIMILARITY_THRESHOLD).astype(int)
        df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

        n_relevant = df["is_relevant"].sum()
        log.info("Similarity computed: %d/%d items above threshold (%.2f)",
                 n_relevant, len(df), SIMILARITY_THRESHOLD)
        return df

    def save_embeddings(self, results: list, output_path: str):
        """
        Save a list of EmbeddingResult objects to disk.
        Saves both:
          - .pkl  (full EmbeddingResult objects, for Python use)
          - .npy  (stacked embedding matrix, for fast numpy loading)
          - .json (identifiers only, for human inspection)
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Pickle (full objects)
        with open(str(out) + ".pkl", "wb") as f:
            pickle.dump(results, f)

        # Numpy matrix
        matrix = np.vstack([r.embedding for r in results])
        np.save(str(out) + ".npy", matrix)

        # JSON metadata
        meta = [{"identifier": r.identifier, "token_count": r.token_count,
                  "was_truncated": r.was_truncated, "language": r.language}
                for r in results]
        with open(str(out) + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info("Embeddings saved → %s.pkl | .npy | _meta.json  (%d items, dim=%d)",
                 out, len(results), matrix.shape[1])

    @staticmethod
    def load_embeddings(output_path: str) -> list:
        """Load previously saved embeddings from .pkl file."""
        with open(str(output_path) + ".pkl", "rb") as f:
            results = pickle.load(f)
        log.info("Loaded %d embeddings from %s", len(results), output_path)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION WITH AST PARSER OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def embed_from_ast_features(
    ast_features_path: str,
    repo_root:         str,
    embedder:          GraphCodeBERTEmbedder,
    language:          str = "python",
) -> list:
    """
    Reads the ast_features.json produced by ast_parser.py and generates
    embeddings for every function/method found.

    This is the main integration point between the AST parser and embedder.

    Returns a list of EmbeddingResult objects, one per function.
    """
    with open(ast_features_path) as f:
        ast_data = json.load(f)

    repo = Path(repo_root)
    snippets    = []
    identifiers = []

    for file_info in ast_data:
        file_path = file_info.get("file_path", "")
        p = Path(file_path)

        if not p.exists():
            continue

        try:
            source_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        # Embed each function individually for fine-grained similarity
        for fn in file_info.get("functions", []) + file_info.get("methods", []):
            start = fn.get("start_line", 1) - 1
            end   = fn.get("end_line", start + 1)
            snippet = "\n".join(source_lines[start:end])

            class_prefix = f"{fn['class_name']}." if fn.get("class_name") else ""
            identifier   = f"{p.name}#{class_prefix}{fn['name']}"

            snippets.append(snippet)
            identifiers.append(identifier)

    log.info("Extracted %d function snippets from AST features", len(snippets))

    if not snippets:
        log.warning("No snippets extracted. Check that source files exist at: %s", repo_root)
        return []

    results = embedder.embed_batch(snippets, identifiers=identifiers, language=language)
    return results


def compute_test_relevance_scores(
    pr_diff:           str,
    test_embeddings:   list,
    embedder:          GraphCodeBERTEmbedder,
    top_k:             int = 50,
) -> pd.DataFrame:
    """
    Given a PR diff and pre-computed test embeddings, compute which tests
    are most semantically relevant to run.

    This is what drives the test selection in the Green-Ops framework:
    only the top_k most similar tests are candidates for execution.

    Args:
        pr_diff:          Unified diff string from the PR
        test_embeddings:  Pre-computed EmbeddingResult list for all tests
        embedder:         Loaded GraphCodeBERTEmbedder instance
        top_k:            Return at most this many test recommendations

    Returns:
        DataFrame with columns: test_name, similarity_score, is_relevant, rank
    """
    if not test_embeddings:
        log.error("No test embeddings provided")
        return pd.DataFrame()

    # Embed the PR diff
    log.info("Embedding PR diff (%d chars) ...", len(pr_diff))
    diff_embedding = embedder.embed_diff(pr_diff)

    # Compare to all test embeddings
    df = embedder.compute_similarity(diff_embedding, test_embeddings)
    df["rank"] = range(1, len(df) + 1)
    df = df.head(top_k)

    log.info("Top-%d tests by semantic similarity to PR diff:", top_k)
    print(df[["rank", "identifier", "similarity_score", "is_relevant"]].head(20).to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Green-Ops GraphCodeBERT Embedder")
    parser.add_argument("--ast-features", default="./greenops_output/ast_features.json",
                        help="Path to ast_features.json from ast_parser.py")
    parser.add_argument("--repo",         default=".",
                        help="Repo root (for reading source files)")
    parser.add_argument("--diff",         default=None,
                        help="Path to a .diff file to embed and rank against tests")
    parser.add_argument("--outdir",       default="./greenops_output")
    parser.add_argument("--language",     default="python", choices=["python", "java"])
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    embedder = GraphCodeBERTEmbedder(device=args.device)
    embedder.load_model()

    # Embed all functions from AST features
    results = embed_from_ast_features(
        ast_features_path = args.ast_features,
        repo_root         = args.repo,
        embedder          = embedder,
        language          = args.language,
    )

    if results:
        embedder.save_embeddings(results, str(out / "graphcodebert_embeddings"))
        print(f"\n✓ Embeddings: {len(results)} functions embedded (dim={EMBEDDING_DIM})")

    # Optional: rank tests against a PR diff
    if args.diff:
        diff_text = Path(args.diff).read_text()
        df_scores = compute_test_relevance_scores(diff_text, results, embedder)
        df_scores.to_csv(out / "test_relevance_scores.csv", index=False)
        print(f"✓ Test relevance scores saved → {out / 'test_relevance_scores.csv'}")

    print(f"\nAll outputs in: {out.resolve()}")
