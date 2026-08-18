"""Retrieval-level evaluation, and the sweep used to tune the threshold.

Relevance rule: a row is relevant to a question if any of its columns contains
any keyword from that question's answer key (case-insensitive substring).

Metrics at the threshold:
  P     fraction of retrieved rows that are relevant
  R     fraction of relevant rows retrieved
  F1    harmonic mean -- what the sweep maximises
  n     rows retrieved
Ranking metrics, independent of the cutoff:
  AUC   ROC-AUC over the full 300-row ranking
  AP    average precision -- better headline under class imbalance

    python -m evaluation.retrieval_eval

LIMITATION: "any keyword in any column" is generous -- the radiology question,
keyed [Radiology, Invalid, Duplicate], marks every Duplicate-claim row relevant
regardless of department. Reported as-is rather than silently tightened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.cases import ANSWER_KEY
from retrieval.embedding import DEFAULT_BACKEND
from retrieval.retriever import DEFAULT_THRESHOLD, Retriever

SWEEP = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def relevance_labels(df: pd.DataFrame, keywords: list[str]) -> pd.Series:
    """True where any column of the row contains any keyword."""
    haystack = df.astype(str).apply(lambda col: col.str.lower()).agg(" ".join, axis=1)
    mask = pd.Series(False, index=df.index)
    for keyword in keywords:
        mask |= haystack.str.contains(keyword.lower(), regex=False)
    return mask


def metrics(scores: np.ndarray, relevant: pd.Series, threshold: float) -> dict:
    retrieved = scores >= threshold * scores.max()
    labels = relevant.to_numpy()

    n_retrieved = int(retrieved.sum())
    n_relevant = int(labels.sum())
    hits = int((retrieved & labels).sum())

    precision = hits / n_retrieved if n_retrieved else 0.0
    recall = hits / n_relevant if n_relevant else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    both = 0 < n_relevant < len(labels)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "n_retrieved": n_retrieved, "n_relevant": n_relevant,
        "roc_auc": roc_auc_score(labels.astype(int), scores) if both else float("nan"),
        "avg_precision": average_precision_score(labels.astype(int), scores) if both else float("nan"),
    }


def sweep(retriever: Retriever) -> float:
    """Score every candidate threshold. Returns the best by mean F1."""
    print(f"\n{'rel. thr':>10}{'P':>8}{'R':>8}{'F1':>8}{'mean rows':>11}")
    print("-" * 78)

    best, best_f1 = DEFAULT_THRESHOLD, -1.0
    scored = {q: retriever.score(q) for q, _ in ANSWER_KEY}

    for candidate in SWEEP:
        rows = [metrics(scored[q], relevance_labels(retriever.df, kw), candidate)
                for q, kw in ANSWER_KEY]
        n = len(rows)
        p = sum(r["precision"] for r in rows) / n
        r_ = sum(r["recall"] for r in rows) / n
        f1 = sum(r["f1"] for r in rows) / n
        size = sum(r["n_retrieved"] for r in rows) / n

        mark = ""
        if f1 > best_f1:
            best, best_f1, mark = candidate, f1, "  <- best"
        print(f"{candidate:>10.2f}{p:>8.2f}{r_:>8.2f}{f1:>8.2f}{size:>11.1f}{mark}")

    print(f"\nBest mean F1 at threshold {best:.2f}. DEFAULT_THRESHOLD is "
          f"{DEFAULT_THRESHOLD:.2f}.")
    return best


def run(threshold: float = DEFAULT_THRESHOLD, backend: str = DEFAULT_BACKEND) -> None:
    print("=" * 78)
    print(f"RETRIEVAL EVALUATION -- backend={backend}, threshold={threshold}")
    print("relevant = row where any column contains any answer-key keyword")
    print("=" * 78)

    retriever = Retriever(backend=backend)
    print(f"{'question':<40}{'rel':>5}{'got':>5}{'P':>7}{'R':>7}{'F1':>7}{'AUC':>7}")
    print("-" * 78)

    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}
    for question, keywords in ANSWER_KEY:
        m = metrics(retriever.score(question),
                    relevance_labels(retriever.df, keywords), threshold)
        for key in totals:
            totals[key] += m[key]
        print(f"{question[:39]:<40}{m['n_relevant']:>5}{m['n_retrieved']:>5}"
              f"{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}"
              f"{m['roc_auc']:>7.3f}")

    n = len(ANSWER_KEY)
    print("-" * 78)
    print(f"{'MEAN':<40}{'':>5}{'':>5}{totals['precision']/n:>7.2f}"
          f"{totals['recall']/n:>7.2f}{totals['f1']/n:>7.2f}{totals['roc_auc']/n:>7.3f}")

    sweep(retriever)
    print("\nThese numbers respond to retrieval quality and cannot be moved by")
    print("rewording the answer string.")


if __name__ == "__main__":
    run()
