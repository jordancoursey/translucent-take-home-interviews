"""Rank claims against a question and keep everything above a threshold.

One retrieval step. No fixed k: the cutoff is a similarity threshold, tuned
against evaluation/retrieval_eval.py rather than guessed.

Why a threshold and not top-k: measured on this dataset the ranking is already
well separated (ROC-AUC 0.89-1.0), so the question is never "which rows rank
highest" but "where does relevant stop". A fixed k answers that with a constant
that is wrong for every question -- 3 is too few for a 37-row department, 50 is
too many. A threshold lets the retrieved set size follow the question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .data_prep import prepare
from .embedding import DEFAULT_BACKEND, Embedder, embed_query, get_embedder

# Keep rows scoring at least this fraction of the best match for the question.
# Tuned by the sweep in evaluation/retrieval_eval.py.
#
# Relative rather than absolute: absolute cutoffs assume every question's scores
# live on the same scale, and they do not. At an absolute 0.40 the cardiology
# question returned 169 rows spanning nine departments -- "denied most often" is
# close to every row in the corpus -- while the radiology question returned a
# clean 30. Scaling to each question's own top score fixes that with one knob.
DEFAULT_THRESHOLD = 0.75


class Retriever:
    """Holds the embedded corpus so it is built once, not once per question."""

    def __init__(self, backend: str = DEFAULT_BACKEND, use_cache: bool = True):
        self.embedder: Embedder = get_embedder(backend)
        self.df, self.documents, self.matrix = prepare(self.embedder, use_cache)

    def score(self, question: str) -> np.ndarray:
        """Similarity of every document to the question. One score per row."""
        return cosine_similarity(
            embed_query(self.embedder, question), self.matrix
        ).flatten()

    def retrieve(
        self, question: str, threshold: float = DEFAULT_THRESHOLD
    ) -> pd.DataFrame:
        """Every row scoring within `threshold` of the best match, best first."""
        scores = self.score(question)
        cutoff = threshold * scores.max()
        keep = np.flatnonzero(scores >= cutoff)
        keep = keep[np.argsort(scores[keep])[::-1]]
        out = self.df.iloc[keep].copy()
        out["_score"] = scores[keep]
        return out


def retrieve(
    question: str, threshold: float = DEFAULT_THRESHOLD, backend: str = DEFAULT_BACKEND
) -> pd.DataFrame:
    """Convenience wrapper. Builds a Retriever per call -- prefer the class."""
    return Retriever(backend=backend).retrieve(question, threshold)
