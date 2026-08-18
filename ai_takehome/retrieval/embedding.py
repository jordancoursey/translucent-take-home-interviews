"""Query and document embedding.

Two backends behind one interface:

  SentenceTransformerEmbedder -- dense semantic vectors (all-MiniLM-L6-v2).
                                 Downloads ~90MB on first use.
  TfidfEmbedder               -- sparse lexical vectors. No download, and what
                                 the original baseline_agent used.

The backend is always stated explicitly and never falls back. An earlier
version resolved "auto" to sentence-transformers or TF-IDF depending on whether
the model happened to load, which meant two runs recorded under the same label
could have used different models -- and a recorded metric you cannot attribute
to a model is not a measurement. If the requested model is unavailable this
raises instead of quietly downgrading.

TF-IDF remains available as the control condition: running both backends over
the same retrieval metrics is how we tell whether dense embeddings help here.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BACKEND = "st"
BACKENDS = ("st", "tfidf")


class Embedder:
    """Interface: fit on a corpus, then encode arbitrary text to a matrix."""

    name = "base"
    model_id = "base"        # concrete model actually used, for result tracking

    def fit(self, documents: list[str]) -> "Embedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Sparse lexical baseline. Matches the original baseline_agent behaviour."""

    name = "tfidf"
    model_id = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vect = TfidfVectorizer()
        self._fitted = False

    def fit(self, documents: list[str]) -> "TfidfEmbedder":
        self._vect.fit(documents)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before encode()")
        return self._vect.transform(texts).toarray()


class SentenceTransformerEmbedder(Embedder):
    """Dense semantic embeddings. Vectors are L2-normalised by the model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.name = f"st:{model_name}"
        self.model_id = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )


def get_embedder(
    backend: str = DEFAULT_BACKEND, model_name: str = DEFAULT_MODEL
) -> Embedder:
    """Build an embedder. Never falls back -- an unavailable model is an error.

    backend="st"     sentence-transformers
    backend="tfidf"  TF-IDF
    """
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "st":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown backend: {backend!r}. Choose one of {BACKENDS}.")


def embed_query(embedder: Embedder, question: str) -> np.ndarray:
    """Encode a single question to a 1 x d matrix."""
    return embedder.encode([question])
