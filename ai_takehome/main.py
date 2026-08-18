"""Agent entry point. Wires retrieval -> generation and runs the evaluators.

    python main.py ask -q "Why are cardiology claims denied most often?"
    python main.py eval-answer
    python main.py eval-retrieval          # includes the threshold sweep
    python main.py eval-all
    python main.py ask -q "..." --threshold 0.4 --backend tfidf
"""

from __future__ import annotations

import argparse
import functools
import os
import pathlib

from generator.generator import active_backend, generate
from observability.tracing import log_trace
from retrieval.embedding import BACKENDS, DEFAULT_BACKEND
from retrieval.retriever import DEFAULT_THRESHOLD, Retriever

def _load_dotenv() -> None:
    """Read .env into the environment if present. No dependency required."""
    path = pathlib.Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# Bump when the pipeline's behaviour changes, so tracked runs stay comparable.
#   0.1.0  TF-IDF, fixed top-3 (baseline_agent)
#   0.2.0  MiniLM embeddings, fixed k
#   0.3.0  relative similarity threshold, single retrieval step
PIPELINE_VERSION = "0.3.0"


@functools.lru_cache(maxsize=4)
def _retriever(backend: str) -> Retriever:
    """One Retriever per backend, reused across questions."""
    return Retriever(backend=backend)


def answer(
    question: str,
    threshold: float = DEFAULT_THRESHOLD,
    backend: str = DEFAULT_BACKEND,
    trace: bool = True,
) -> str:
    """Full pipeline: embed -> retrieve above threshold -> synthesise.

    Every call is appended to the trace log; pass trace=False to skip it.
    """
    retriever = _retriever(backend)
    rows = retriever.retrieve(question, threshold=threshold)
    text = generate(question, rows)
    if trace:
        log_trace(
            query=question,
            retrieved_row_ids=rows.index,
            answer=text,
            retrieval_model=retriever.embedder.model_id,
            generator_model=active_backend(),
            pipeline_version=PIPELINE_VERSION,
            threshold=threshold,
        )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="ask",
                        choices=["ask", "eval-answer", "eval-retrieval", "eval-all"])
    parser.add_argument("--question", "-q")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"similarity cutoff (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(BACKENDS))
    args = parser.parse_args()

    if args.command == "ask":
        if not args.question:
            parser.error("ask requires --question")
        print(answer(args.question, threshold=args.threshold, backend=args.backend))
        return

    if args.command in ("eval-answer", "eval-all"):
        from evaluation import answer_eval
        answer_eval.run(threshold=args.threshold, backend=args.backend)

    if args.command in ("eval-retrieval", "eval-all"):
        from evaluation import retrieval_eval
        retrieval_eval.run(threshold=args.threshold, backend=args.backend)


if __name__ == "__main__":
    main()
