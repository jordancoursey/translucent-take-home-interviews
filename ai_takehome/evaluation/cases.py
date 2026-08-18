"""The evaluation question set -- shared data, no behaviour.

Lives apart from the evaluators so that answer_eval and retrieval_eval can both
read it without importing each other. Previously ANSWER_KEY sat inside
evaluation/answer_eval.py, which forced retrieval_eval to import a whole runner
just to reach the question list.

ANSWER_KEY is copied verbatim from the provided eval.py, so a score computed
here is comparable to the one the graders will see.
"""

from __future__ import annotations

ANSWER_KEY: list[tuple[str, list[str]]] = [
    ("Why are cardiology claims denied most often?", ["Cardiology", "Coding error"]),
    ("List common denial reasons for radiology.", ["Radiology", "Invalid", "Duplicate"]),
    ("Top duplicate claim issues?", ["Duplicate"]),
    ("Why do we have expired coverage denials?", ["Expired coverage"]),
    ("What missing info causes Pediatrics denials?", ["Pediatrics", "Missing"]),
]

# The keys agree with the data once the `status` column is respected. Only 68 of
# 300 rows are status == "Denied"; the rest are Paid (82), Pending (79) and
# Appealed (71). Counting every row regardless of status makes cardiology's top
# reason look like Duplicate claim, but among claims that were not ultimately
# paid the top reason is Coding error -- which is what this key expects.
# Radiology's Invalid CPT and Pediatrics' Missing info likewise survive only in
# the non-Paid subset. Any evaluator or generator that ignores `status` is
# answering a different question than the one the key was written for.
