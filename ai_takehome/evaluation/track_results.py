"""Record per-question precision and recall so runs can be compared over time.

Appends one JSON object per question per run to results/metrics.jsonl, and
overwrites results/metrics_latest.csv with just the most recent run.

Two layers are recorded for each of the five evaluation questions:

  retrieval_*  precision/recall of the rows retrieved, against the relevance
               rule in evaluation/retrieval_eval.py
  answer_*     precision/recall of the phrases in the generated answer, against
               the answer key in evaluation/cases.py

    python -m evaluation.track_results
    python -m evaluation.track_results --threshold 0.8 --backend tfidf --note "wider cutoff"
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from datetime import datetime, timezone

from evaluation import answer_eval, retrieval_eval
from retrieval.embedding import BACKENDS, DEFAULT_BACKEND
from retrieval.retriever import DEFAULT_THRESHOLD, Retriever

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
JSONL_PATH = RESULTS_DIR / "metrics.jsonl"
CSV_PATH = RESULTS_DIR / "metrics_latest.csv"

FIELDS = [
    "run_id", "timestamp", "version", "model", "backend", "threshold", "note",
    "question",
    "retrieval_n_relevant", "retrieval_n_retrieved",
    "retrieval_precision", "retrieval_recall", "retrieval_f1",
    "retrieval_roc_auc", "retrieval_avg_precision",
    "answer_passed", "answer_recall", "answer_precision_key",
    "answer_precision_true", "answer_text",
]


def collect(threshold: float, backend: str, note: str) -> list[dict]:
    """One record per evaluation question."""
    import main

    retriever = Retriever(backend=backend)
    df = retriever.df
    # `backend` is the requested family; `model` is the concrete model used.
    model = retriever.embedder.model_id
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = timestamp.replace(":", "").replace("-", "")

    records = []
    for question, keywords in answer_eval.ANSWER_KEY:
        relevant = retrieval_eval.relevance_labels(df, keywords)
        rm = retrieval_eval.metrics(retriever.score(question), relevant, threshold)

        text = main.answer(question, threshold=threshold, backend=backend)
        passed, missing = answer_eval.score_answer(text, keywords)
        p_key, p_true = answer_eval.precision(text, keywords, df)

        records.append({
            "run_id": run_id,
            "timestamp": timestamp,
            "version": main.PIPELINE_VERSION,
            "model": model,
            "backend": backend,
            "threshold": threshold,
            "note": note,
            "question": question,
            "retrieval_n_relevant": rm["n_relevant"],
            "retrieval_n_retrieved": rm["n_retrieved"],
            "retrieval_precision": round(rm["precision"], 4),
            "retrieval_recall": round(rm["recall"], 4),
            "retrieval_f1": round(rm["f1"], 4),
            "retrieval_roc_auc": round(rm["roc_auc"], 4),
            "retrieval_avg_precision": round(rm["avg_precision"], 4),
            "answer_passed": int(passed),
            "answer_recall": round((len(keywords) - len(missing)) / len(keywords), 4),
            "answer_precision_key": round(p_key, 4),
            "answer_precision_true": round(p_true, 4),
            "answer_text": text,
        })
    return records


def display_path(path: pathlib.Path) -> str:
    """Relative to cwd when possible; absolute otherwise."""
    try:
        return str(path.relative_to(pathlib.Path.cwd()))
    except ValueError:
        return str(path)


def write(records: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with JSONL_PATH.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def report(records: list[dict]) -> None:
    print(f"{'question':<42}{'retP':>6}{'retR':>6}{'ansP':>6}{'ansR':>6}{'pass':>6}")
    print("-" * 72)
    for r in records:
        print(f"{r['question'][:41]:<42}"
              f"{r['retrieval_precision']:>6.2f}{r['retrieval_recall']:>6.2f}"
              f"{r['answer_precision_true']:>6.2f}{r['answer_recall']:>6.2f}"
              f"{'Y' if r['answer_passed'] else 'N':>6}")

    n = len(records)
    print("-" * 72)
    print(f"{'MEAN':<42}"
          f"{sum(r['retrieval_precision'] for r in records)/n:>6.2f}"
          f"{sum(r['retrieval_recall'] for r in records)/n:>6.2f}"
          f"{sum(r['answer_precision_true'] for r in records)/n:>6.2f}"
          f"{sum(r['answer_recall'] for r in records)/n:>6.2f}"
          f"{sum(r['answer_passed'] for r in records):>5}/{n}")


def history(limit: int = 10) -> None:
    """Mean scores per past run, oldest first."""
    if not JSONL_PATH.exists():
        return
    runs: dict[str, list[dict]] = {}
    for line in JSONL_PATH.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            runs.setdefault(record["run_id"], []).append(record)

    print(f"\n{'run':<18}{'ver':>7}{'model':>20}{'thr':>6}{'retP':>6}{'retR':>6}"
          f"{'ansR':>6}{'pass':>6}  note")
    print("-" * 96)
    for run_id in sorted(runs)[-limit:]:
        rows = runs[run_id]
        n = len(rows)
        print(f"{run_id[:17]:<18}{rows[0].get('version', '-'):>7}"
              f"{rows[0].get('model', rows[0].get('backend', '-')):>20}"
              f"{rows[0]['threshold']:>6.2f}"
              f"{sum(r['retrieval_precision'] for r in rows)/n:>6.2f}"
              f"{sum(r['retrieval_recall'] for r in rows)/n:>6.2f}"
              f"{sum(r['answer_recall'] for r in rows)/n:>6.2f}"
              f"{sum(r['answer_passed'] for r in rows):>4}/{n}  {rows[0]['note']}")


def main_() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(BACKENDS))
    parser.add_argument("--note", default="", help="label this run in the history")
    args = parser.parse_args()

    records = collect(args.threshold, args.backend, args.note)
    write(records)
    report(records)
    history()
    print(f"\nappended to {display_path(JSONL_PATH)}"
          f"  ({len(records)} rows)")
    print(f"latest run   {display_path(CSV_PATH)}")


if __name__ == "__main__":
    main_()
