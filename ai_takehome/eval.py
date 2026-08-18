"""Provided evaluation, extended to compare the baseline against the new pipeline.

The five tests and the scoring rule are unchanged from the version shipped with
the exercise: a question passes only if every expected keyword appears in the
answer, case-insensitively. Both systems are scored by that same rule so the
numbers are directly comparable.

  baseline  baseline_agent.answer -- TF-IDF, fixed top-3 (as provided, plus the
            service_date fix needed to make it run at all)
  pipeline  main.answer -- MiniLM embeddings, relative similarity threshold,
            aggregate-and-synthesise generator

    python eval.py
    python eval.py --verbose     # also print every answer, with missing keywords
"""

import argparse

from baseline_agent import answer as baseline_answer

tests = [
    ("Why are cardiology claims denied most often?", ["Cardiology", "Coding error"]),
    ("List common denial reasons for radiology.", ["Radiology", "Invalid", "Duplicate"]),
    ("Top duplicate claim issues?", ["Duplicate"]),
    ("Why do we have expired coverage denials?", ["Expired coverage"]),
    ("What missing info causes Pediatrics denials?", ["Pediatrics", "Missing"])
]


def score(answer_fn):
    """The original scoring loop, plus the text and any missing keywords.

    Returns (passed, [(ok, answer, missing), ...]). The pass/fail decision is
    unchanged -- `missing` is only recorded so --verbose can show why.
    """
    outcomes = []
    for q, keywords in tests:
        text = answer_fn(q)
        low = text.lower()
        missing = [k for k in keywords if k.lower() not in low]
        outcomes.append((not missing, text, missing))
    return sum(ok for ok, _, _ in outcomes), outcomes


def load_pipeline():
    """The new pipeline, or None if its model is unavailable.

    Kept optional so eval.py still runs on a machine that cannot download the
    sentence-transformers model.
    """
    try:
        from main import answer
        answer("warmup", trace=False)
        return answer
    except Exception as exc:
        print(f"[pipeline unavailable: {type(exc).__name__}: {exc}]\n")
        return None


def report_verbose(baseline_outcomes, pipeline_outcomes):
    """Print both answers per question, with the keywords each one missed."""
    for i, ((q, keywords), base, pipe) in enumerate(
        zip(tests, baseline_outcomes, pipeline_outcomes), start=1
    ):
        print(f"\n{'=' * 74}\nQ{i}. {q}\n     expected keywords: {keywords}\n{'=' * 74}")
        for label, (ok, text, missing) in (("baseline", base), ("pipeline", pipe)):
            print(f"\n  {label}  [{'PASS' if ok else 'FAIL'}]"
                  + (f"  missing: {missing}" if missing else ""))
            print(f"    {text}")
    print()


parser = argparse.ArgumentParser(description="Score the baseline and the pipeline.")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="print each system's answer per question")
args = parser.parse_args()

baseline_passed, baseline_outcomes = score(baseline_answer)
pipeline_answer = load_pipeline()

if pipeline_answer is None:
    if args.verbose:
        for i, ((q, keywords), (ok, text, missing)) in enumerate(
            zip(tests, baseline_outcomes), start=1
        ):
            print(f"\nQ{i}. {q}\n  baseline  [{'PASS' if ok else 'FAIL'}]"
                  + (f"  missing: {missing}" if missing else ""))
            print(f"    {text}")
        print()
    print(f"Score: {baseline_passed}/{len(tests)}")
else:
    pipeline_passed, pipeline_outcomes = score(pipeline_answer)

    if args.verbose:
        report_verbose(baseline_outcomes, pipeline_outcomes)

    print(f"{'question':<44}{'baseline':>10}{'pipeline':>10}")
    print("-" * 64)
    for (q, _), base, pipe in zip(tests, baseline_outcomes, pipeline_outcomes):
        mark = lambda ok: "PASS" if ok else "FAIL"
        print(f"{q[:43]:<44}{mark(base[0]):>10}{mark(pipe[0]):>10}")
    print("-" * 64)
    print(f"{'Score':<44}{f'{baseline_passed}/{len(tests)}':>10}"
          f"{f'{pipeline_passed}/{len(tests)}':>10}")

    print(f"\nScore: {pipeline_passed}/{len(tests)}")
