"""Load the claims CSV, turn rows into documents, and embed the corpus once.

The baseline re-read the CSV and re-fit the vectorizer inside every answer()
call. Corpus embedding happens here instead, once, with an on-disk cache keyed
by backend name.

Document text includes every field that a question might reference -- the
baseline's template dropped payer, amount, status and age, so no question about
them could ever retrieve correctly.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pandas as pd

from .embedding import Embedder

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "denials.csv"
CACHE_DIR = DATA_PATH.parent / ".embedding_cache"


def load_dataframe(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def row_to_document(row: pd.Series) -> str:
    """Render one claim as retrievable text.

    Phrased so that both a department question and a denial-reason question
    have lexical and semantic overlap with it.
    """
    return (
        f"Department {row['department']} denied claim {row['claim_id']} "
        f"because {row['denial_reason']}. "
        f"Payer {row['payer']}, amount ${row['amount']}, "
        f"status {row['status']}, patient age {row['patient_age']}, "
        f"service date {row['service_date']}."
    )


def build_corpus(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = load_dataframe() if df is None else df
    return df, df.apply(row_to_document, axis=1).tolist()


def _cache_path(embedder_name: str, documents: list[str]) -> pathlib.Path:
    digest = hashlib.sha256(
        (embedder_name + "\n".join(documents)).encode()
    ).hexdigest()[:16]
    safe = embedder_name.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.{digest}.npy"


def embed_corpus(
    embedder: Embedder,
    documents: list[str],
    use_cache: bool = True,
) -> np.ndarray:
    """Embed every document. Cached on disk; invalidated if documents change."""
    embedder.fit(documents)

    path = _cache_path(embedder.name, documents)
    if use_cache and path.exists():
        return np.load(path)

    matrix = embedder.encode(documents)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
    return matrix


def prepare(embedder: Embedder, use_cache: bool = True):
    """One call to get everything downstream needs."""
    df, documents = build_corpus()
    matrix = embed_corpus(embedder, documents, use_cache=use_cache)
    return df, documents, matrix
