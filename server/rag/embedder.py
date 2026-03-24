"""Embedder — Section 9, Embedding Strategy.

Dense: all-MiniLM-L6-v2 from sentence-transformers (384-dim, cosine).
Sparse: BM25 term frequency weighting with IDF on tokenized embed_text.
Only embed_text is embedded. Body is never embedded.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("server.embedder")

# Sparse vector vocab size: hash(term) mod 2^20 per Section 9
SPARSE_VOCAB_SIZE = 2**20


class Embedder:
    """Produces dense and sparse vectors from embed_text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name, local_files_only=True)
        self._k1 = 1.5   # BM25 parameter
        self._b = 0.75    # BM25 parameter
        self._avgdl = 50.0  # Average document length estimate

        # IDF statistics — updated as documents are indexed
        self._lock = threading.Lock()
        self._doc_count = 0
        self._term_doc_freq: Counter = Counter()  # term -> number of docs containing term

    # ------------------------------------------------------------------
    # Corpus statistics for IDF
    # ------------------------------------------------------------------

    def update_idf_stats(self, texts: list[str]) -> None:
        """Update IDF statistics from a batch of indexed documents.

        Called during /index to accumulate corpus term frequencies.
        Thread-safe via lock.
        """
        with self._lock:
            for text in texts:
                tokens = set(self._tokenize(text))  # unique terms per doc
                for token in tokens:
                    self._term_doc_freq[token] += 1
                self._doc_count += 1
            # Update avgdl estimate
            total_tokens = sum(len(self._tokenize(t)) for t in texts)
            if self._doc_count > 0:
                self._avgdl = max(1.0, (self._avgdl * (self._doc_count - len(texts)) + total_tokens) / self._doc_count)

    def _idf(self, term: str) -> float:
        """Compute IDF for a term. Returns 1.0 if no corpus stats available."""
        with self._lock:
            if self._doc_count == 0:
                return 1.0
            df = self._term_doc_freq.get(term, 0)
            if df == 0:
                return 1.0
            # Standard BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            return math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)

    # ------------------------------------------------------------------
    # Dense embedding
    # ------------------------------------------------------------------

    def embed_dense(self, text: str) -> list[float]:
        """Produce a 384-dimensional L2-normalized dense vector."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch dense embedding."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # Sparse embedding (BM25 with IDF)
    # ------------------------------------------------------------------

    def embed_sparse(self, text: str) -> dict[int, float]:
        """Produce a sparse BM25 vector with IDF weighting.

        Term indices = hash(term) mod 2^20.
        Values = BM25 IDF * TF score.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        tf = Counter(tokens)
        dl = len(tokens)

        sparse: dict[int, float] = {}
        for term, count in tf.items():
            # BM25 TF component
            tf_score = (count * (self._k1 + 1)) / (
                count + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            )
            # IDF weighting — improves sparse search quality
            idf = self._idf(term)
            idx = self._term_index(term)
            sparse[idx] = idf * tf_score

        return sparse

    def embed_both(self, text: str) -> tuple[list[float], dict[int, float]]:
        """Produce both dense and sparse vectors."""
        return self.embed_dense(text), self.embed_sparse(text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if t]

    @staticmethod
    def _term_index(term: str) -> int:
        """Compute term index as hash(term) mod 2^20."""
        h = int(hashlib.md5(term.encode()).hexdigest(), 16)
        return h % SPARSE_VOCAB_SIZE
