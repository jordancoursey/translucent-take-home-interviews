"""Append-only trace log for every question the pipeline answers.

One CSV row per call:

    trace_id            uuid4, unique per call
    timestamp           UTC, ISO-8601
    query               the user's question, verbatim
    retrieved_row_ids   dataframe row numbers that retrieval returned,
                        space-separated, best match first
    answer              the text returned to the user
    retrieval_model     concrete embedding model, e.g. all-MiniLM-L6-v2
    generator_model     what composed the answer, e.g. deterministic-template
    pipeline_version    PIPELINE_VERSION at the time of the call
    threshold           relative similarity cutoff used for this call

An answer you cannot attribute to a model and a cutoff is not reproducible, so
the identity of both stages travels with every trace rather than being inferred
from whatever the code happens to say today.

Append-only by construction: the file is opened "a" and one row is written per
call. Nothing here reads-modifies-writes, so a trace cannot be edited or lost
by a later run -- which is the point of a trace log. The header is written only
when the file does not yet exist.
"""

from __future__ import annotations

import csv
import pathlib
import threading
import uuid
from datetime import datetime, timezone

TRACE_DIR = pathlib.Path(__file__).resolve().parent / "traces"
TRACE_PATH = TRACE_DIR / "traces.csv"

FIELDS = [
    "trace_id", "timestamp", "query", "retrieved_row_ids", "answer",
    "retrieval_model", "generator_model", "pipeline_version", "threshold",
]

# Guards the append so concurrent callers in one process cannot interleave rows.
_LOCK = threading.Lock()


def _format_ids(row_ids) -> str:
    """Row numbers as a space-separated string, order preserved."""
    if row_ids is None:
        return ""
    return " ".join(str(int(i)) for i in row_ids)


def _existing_header(path: pathlib.Path) -> list[str] | None:
    """The header row already on disk, or None if the file is new/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as handle:
        return next(csv.reader(handle), None)


def _rotate(path: pathlib.Path) -> pathlib.Path:
    """Move a file written under an older schema aside, keeping its rows.

    Appending new columns to an existing CSV would misalign every prior row
    against the header. Rotating preserves the old traces intact instead of
    rewriting or discarding them -- append-only holds within each file.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archived = path.with_name(f"{path.stem}.{stamp}.csv")
    path.rename(archived)
    return archived


def log_trace(
    query: str,
    retrieved_row_ids,
    answer: str,
    retrieval_model: str = "",
    generator_model: str = "",
    pipeline_version: str = "",
    threshold: float | None = None,
    path: pathlib.Path = TRACE_PATH,
) -> str:
    """Append one trace row. Returns the trace_id."""
    trace_id = str(uuid.uuid4())
    row = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": query,
        "retrieved_row_ids": _format_ids(retrieved_row_ids),
        "answer": answer,
        "retrieval_model": retrieval_model,
        "generator_model": generator_model,
        "pipeline_version": pipeline_version,
        "threshold": "" if threshold is None else threshold,
    }

    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = _existing_header(path)
        if header is not None and header != FIELDS:
            archived = _rotate(path)
            print(f"[tracing] trace schema changed; previous log kept at {archived.name}")
            header = None

        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if header is None:
                writer.writeheader()
            writer.writerow(row)

    return trace_id


def read_traces(path: pathlib.Path = TRACE_PATH) -> list[dict]:
    """Every trace, oldest first. Read-only helper for inspection."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
