"""Demonstrates that eval.py's scoring is gameable.

eval.py grades with `all(keyword in answer)` -- pure recall of hand-written key
phrases. There is no penalty for extra words, no check that the reported counts
are correct, and no visibility into which rows were retrieved.

So an `answer()` that ignores its argument and returns one constant string
scores 5/5, while the real retrieval pipeline scores 2/5.

Run:  python -m evaluation.cheated_eval
"""

from baseline_agent import answer as baseline_answer

# Copied verbatim from eval.py so this scores identically.
TESTS = [
    ("Why are cardiology claims denied most often?", ["Cardiology", "Coding error"]),
    ("List common denial reasons for radiology.", ["Radiology", "Invalid", "Duplicate"]),
    ("Top duplicate claim issues?", ["Duplicate"]),
    ("Why do we have expired coverage denials?", ["Expired coverage"]),
    ("What missing info causes Pediatrics denials?", ["Pediatrics", "Missing"]),
]

# Every phrase the answer key looks for, concatenated. Nothing else.
ALL_KEYWORDS = " ".join(sorted({k for _, kws in TESTS for k in kws}))


def cheated_answer(question: str) -> str:
    """Ignores the question. Reads no data. Retrieves nothing. Scores 5/5."""
    return ALL_KEYWORDS


def score(answer_fn) -> int:
    """The exact scoring loop from eval.py."""
    passed = 0
    for question, keywords in TESTS:
        ans = answer_fn(question).lower()
        if all(k.lower() in ans for k in keywords):
            passed += 1
    return passed


if __name__ == "__main__":
    print(f"cheated_answer returns this for every question:\n  {ALL_KEYWORDS!r}\n")
    print(f"  cheated_answer (no data, no retrieval) : {score(cheated_answer)}/5")
    print(f"  baseline_agent.answer (retrieval agent)  : {score(baseline_answer)}/5")
    print("\nThe hardcoded string outscores the working pipeline, so eval.py cannot")
    print("distinguish retrieval quality from keyword density.")
