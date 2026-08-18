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
"""

from baseline_agent import answer as baseline_answer

tests = [
    ("Why are cardiology claims denied most often?", ["Cardiology", "Coding error"]),
    ("List common denial reasons for radiology.", ["Radiology", "Invalid", "Duplicate"]),
    ("Top duplicate claim issues?", ["Duplicate"]),
    ("Why do we have expired coverage denials?", ["Expired coverage"]),
    ("What missing info causes Pediatrics denials?", ["Pediatrics", "Missing"])
]


def score(answer_fn):
    """The original scoring loop. Returns (passed, per-question outcomes)."""
    outcomes = []
    for q, keywords in tests:
        ans = answer_fn(q).lower()
        outcomes.append(all(k.lower() in ans for k in keywords))
    return sum(outcomes), outcomes


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


baseline_passed, baseline_outcomes = score(baseline_answer)
pipeline_answer = load_pipeline()

if pipeline_answer is None:
    print(f"Score: {baseline_passed}/{len(tests)}")
else:
    pipeline_passed, pipeline_outcomes = score(pipeline_answer)

    print(f"{'question':<44}{'baseline':>10}{'pipeline':>10}")
    print("-" * 64)
    for (q, _), base_ok, pipe_ok in zip(tests, baseline_outcomes, pipeline_outcomes):
        mark = lambda ok: "PASS" if ok else "FAIL"
        print(f"{q[:43]:<44}{mark(base_ok):>10}{mark(pipe_ok):>10}")
    print("-" * 64)
    print(f"{'Score':<44}{f'{baseline_passed}/{len(tests)}':>10}"
          f"{f'{pipeline_passed}/{len(tests)}':>10}")

    print(f"\nScore: {pipeline_passed}/{len(tests)}")
