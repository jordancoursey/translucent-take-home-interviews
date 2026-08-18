"""Answer-level evaluation: does the generated text contain the expected phrases?

Scoring rule (same as the provided eval.py): a question passes only if every
keyword in its answer key appears in the answer, case-insensitively.

Also reports per-question keyword recall, so a near-miss is visible instead of
being flattened to a fail. Note that a perfect score here does not mean the
answers are true -- keyword containment cannot check a count. evaluation/retrieval_eval.py
and the counts in generator/generator.py are what check correctness.

The question set lives in evaluation/cases.py, which also documents how the key
relates to the `status` column.

    python -m evaluation.answer_eval
"""

from __future__ import annotations

from evaluation.cases import ANSWER_KEY


def score_answer(answer: str, keywords: list[str]) -> tuple[bool, list[str]]:
    low = answer.lower()
    missing = [k for k in keywords if k.lower() not in low]
    return not missing, missing


def _subject_rows(df, keywords: list[str]):
    """Rows the question is about: its department if the key names one, else all."""
    departments = set(df.department.unique())
    named = [k for k in keywords if k in departments]
    return df[df.department.isin(named)] if named else df


def precision(answer: str, keywords: list[str], df) -> tuple[float, float]:
    """Two ways of deciding which emitted words count as extraneous.

    p_key   extraneous = not in the answer key. Punishes any word the key does
            not list, including true ones, so a complete breakdown scores badly.
    p_true  extraneous = not true of the rows the question is about. A reason
            counts if it actually occurs in those rows; a department counts if
            it is the subject. Rewards completeness, punishes irrelevance.
    """
    low = answer.lower()
    vocabulary = sorted(set(df.department.unique()) | set(df.denial_reason.unique()))
    emitted = [term for term in vocabulary if term.lower() in low]
    if not emitted:
        return 0.0, 0.0

    on_key = [
        t for t in emitted
        if any(k.lower() in t.lower() or t.lower() in k.lower() for k in keywords)
    ]

    subject = _subject_rows(df, keywords)
    true_reasons = set(subject.denial_reason.unique())
    true_departments = set(subject.department.unique())
    on_true = [t for t in emitted if t in true_reasons or t in true_departments]

    return len(on_key) / len(emitted), len(on_true) / len(emitted)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(answer_fn, label: str, df, verbose: bool = True) -> dict:
    if verbose:
        print(f"\n{label}")
        print("-" * 78)

    passed = 0
    sums = {"recall": 0.0, "p_key": 0.0, "p_true": 0.0}
    for question, keywords in ANSWER_KEY:
        text = answer_fn(question)
        ok, missing = score_answer(text, keywords)
        passed += ok

        recall = (len(keywords) - len(missing)) / len(keywords)
        p_key, p_true = precision(text, keywords, df)
        sums["recall"] += recall
        sums["p_key"] += p_key
        sums["p_true"] += p_true

        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {question}")
            print(f"         recall {recall:.0%}   P_key {p_key:.2f}   P_true {p_true:.2f}"
                  + (f"   missing: {missing}" if missing else ""))
            print(f"         -> {text[:130]}{'...' if len(text) > 130 else ''}")

    n = len(ANSWER_KEY)
    result = {
        "label": label,
        "score": passed,
        "recall": sums["recall"] / n,
        "p_key": sums["p_key"] / n,
        "p_true": sums["p_true"] / n,
    }
    result["f1_key"] = _f1(result["p_key"], result["recall"])
    result["f1_true"] = _f1(result["p_true"], result["recall"])

    if verbose:
        print(f"\n  Score: {passed}/{n}   mean recall {result['recall']:.2f}   "
              f"P_key {result['p_key']:.2f}   P_true {result['p_true']:.2f}")
    return result


def run(threshold: float | None = None, backend: str | None = None) -> None:
    from retrieval.embedding import DEFAULT_BACKEND
    from retrieval.retriever import DEFAULT_THRESHOLD
    backend = DEFAULT_BACKEND if backend is None else backend
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    print("=" * 78)
    print("ANSWER EVALUATION -- keyword containment, plus precision")
    print("=" * 78)

    from retrieval.data_prep import load_dataframe
    df = load_dataframe()

    import main
    results = [evaluate(lambda q: main.answer(q, threshold=threshold, backend=backend),
                        f"pipeline (backend={backend}, t={threshold})", df)]

    try:
        from baseline_agent import answer as baseline
        results.append(evaluate(baseline, "baseline_agent (TF-IDF, top-3)", df))
    except Exception as exc:
        print(f"\n  baseline_agent unavailable: {exc}")

    try:
        from evaluation.cheated_eval import cheated_answer
        results.append(evaluate(cheated_answer, "cheated_answer (constant string)", df))
    except Exception as exc:
        print(f"\n  cheated_answer unavailable: {exc}")

    print("\n" + "=" * 78)
    print(f"{'system':<40}{'score':>7}{'recall':>8}{'P_key':>7}{'P_true':>8}"
          f"{'F1_key':>8}{'F1_true':>9}")
    print("-" * 78)
    for r in results:
        print(f"{r['label'][:39]:<40}{r['score']:>5}/5{r['recall']:>8.2f}"
              f"{r['p_key']:>7.2f}{r['p_true']:>8.2f}{r['f1_key']:>8.2f}{r['f1_true']:>9.2f}")
    print("\nP_key punishes every word outside the answer key, so it penalises a")
    print("complete breakdown. P_true punishes only words untrue of the subject")
    print("rows. Compare how each ranks the constant string against the pipeline.")


if __name__ == "__main__":
    run()
