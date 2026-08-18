"""Turn retrieved claims into an answer.

Reports every reason present in the retrieved set -- no cap. How much appears
in the answer is set by the retrieval threshold, not by a second constant here.

The baseline did value_counts() over 3 rows and joined with pipes, which made
every count a fraction of 3: "Duplicate claim: 1" for a department where
duplicates lead at 6 of 37. Stating the denominator keeps that visible.

No LLM: the README makes it optional, and a template keeps counts reproducible.
"""

from __future__ import annotations

import pandas as pd

# Named in traces so an answer can be attributed to the thing that wrote it.
# Deterministic template rather than a model, hence no weights to version --
# its behaviour is versioned by PIPELINE_VERSION in main.py.
GENERATOR_ID = "deterministic-template"


def _dominant(rows: pd.DataFrame, column: str, threshold: float = 0.5) -> str | None:
    """The single value of `column` covering most of the rows, if there is one."""
    if rows.empty:
        return None
    share = rows[column].value_counts(normalize=True)
    return share.index[0] if share.iloc[0] >= threshold else None


def generate(question: str, rows: pd.DataFrame) -> str:
    """Compose an answer naming the subject, the ranked reasons, and counts."""
    if rows.empty:
        return "No matching claims found."

    total = len(rows)
    department = _dominant(rows, "department")
    counts = rows["denial_reason"].value_counts()

    subject = f"{department} claims" if department else "Matching claims"
    lead_reason, lead_count = counts.index[0], int(counts.iloc[0])

    parts = [
        f"{subject}: {total} denials retrieved. "
        f"The most common reason is {lead_reason} "
        f"({lead_count} of {total}, {100 * lead_count / total:.0f}%)."
    ]

    if len(counts) > 1:
        breakdown = ", ".join(f"{r} {int(c)}" for r, c in counts.items())
        parts.append(f"Breakdown by reason: {breakdown}.")

    if department is None:
        departments = rows["department"].value_counts()
        parts.append(
            "Departments involved: "
            + ", ".join(f"{d} {int(c)}" for d, c in departments.items())
            + "."
        )

    return " ".join(parts)
