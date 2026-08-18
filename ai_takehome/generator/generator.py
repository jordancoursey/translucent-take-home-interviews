"""Turn retrieved claims into an answer.

Two generators, tried in order (see `active_backend`):

  1. Claude via the Anthropic API -- when a credential is present. Receives the
     retrieved claims whole and does its own reasoning.
  2. a deterministic template     -- always available; aggregates the retrieved
     rows and renders them, so its counts are exact by construction.

Whichever ran is recorded in the trace, so an answer is never ambiguous about
its own provenance. A local open-source generator was tried as a middle tier and
removed: it produced fluent prose with no counts or citations, scored 3/5 against
the template's 5/5, and ran ~60x slower.


Relevance-aware in two concrete ways, neither of which invents a confidence number:

  consensus filter  retrieval returns a similarity window, which can straddle the
                    edge of the subject. When the retrieved rows agree on a
                    department, only that department is counted. Department
                    identity is an exact categorical fact; cosine similarity is an
                    uncalibrated float, so the filter is built on the former. When
                    there is no dominant department the question is genuinely
                    cross-department and no filter is applied.

  comparison mode   a question naming two or more departments is answered per
                    department rather than collapsed to the majority. Without
                    this, "compare cardiology and radiology" retrieved both in
                    full and then discarded the 29 Radiology rows because
                    Cardiology held 56% -- answering half the question silently.

  scores order      `_score` is used only to choose which claims to cite as
                    evidence -- ordering is what similarity is legitimately good
                    for. It never weights a count.

The question is checked against the retrieved subject so an off-target retrieval
is reported rather than answered.

The baseline did value_counts() over 3 rows and joined with pipes, which made
every count a fraction of 3: "Duplicate claim: 1" for a department where
duplicates lead at 6 of 37. Stating the denominator keeps that visible.

No LLM: the README makes it optional, and a template keeps counts reproducible.
"""

from __future__ import annotations

import pandas as pd

from generator import llm

# Named in traces so an answer can be attributed to the thing that wrote it.
# Deterministic template rather than a model, hence no weights to version --
# its behaviour is versioned by PIPELINE_VERSION in main.py.
TEMPLATE_ID = "deterministic-template"


def active_backend() -> str:
    """Which generator will run, without invoking it. Used for tracing."""
    return llm.MODEL if llm.available() else TEMPLATE_ID


# Kept for callers that just want a label; prefer active_backend().
GENERATOR_ID = TEMPLATE_ID

CONSENSUS = 0.5      # share of rows that must agree before a subject is claimed
N_CITATIONS = 3      # example claim ids shown as evidence
N_COMPARE_REASONS = 3  # reasons listed per department when comparing several


def _dominant(rows: pd.DataFrame, column: str, threshold: float = CONSENSUS) -> str | None:
    """The single value of `column` covering most of the rows, if there is one."""
    if rows.empty:
        return None
    share = rows[column].value_counts(normalize=True)
    return share.index[0] if share.iloc[0] >= threshold else None


def apply_consensus(rows: pd.DataFrame) -> tuple[pd.DataFrame, str | None, int]:
    """Drop rows dissenting from the majority department.

    Returns (kept rows, department or None, number dropped). A similarity
    threshold is a window, not a boundary, so it can admit a neighbour from
    another department -- for the cardiology question it admitted exactly one
    Radiology row, the lowest-scoring of the 38, which shifted the reported top
    count from the true 6 to 7. Counting only the agreed subject fixes that
    without consulting the score.
    """
    department = _dominant(rows, "department")
    if department is None:
        return rows, None, 0
    kept = rows[rows["department"] == department]
    return kept, department, len(rows) - len(kept)


def _citations(rows: pd.DataFrame, reason: str, limit: int = N_CITATIONS) -> list[str]:
    """Claim ids for `reason`, most relevant first when scores are available."""
    matching = rows[rows["denial_reason"] == reason]
    if "_score" in matching.columns:
        matching = matching.sort_values("_score", ascending=False)
    return matching["claim_id"].head(limit).tolist()


def _off_target(question: str, department: str | None, rows: pd.DataFrame) -> str | None:
    """Warn when the question names a department the retrieved rows are not about."""
    named = {d for d in rows["department"].unique()}
    asked = {d for d in _KNOWN_DEPARTMENTS if d.lower() in question.lower()}
    if not asked:
        return None
    if department is not None and department not in asked:
        return (f"Note: the question mentions {', '.join(sorted(asked))}, but the "
                f"retrieved claims are mostly {department}.")
    if department is None and not (asked & named):
        return (f"Note: no retrieved claim belongs to {', '.join(sorted(asked))}.")
    return None


_KNOWN_DEPARTMENTS = (
    "Cardiology", "Dermatology", "Gastroenterology", "Neurology", "Oncology",
    "Ophthalmology", "Orthopedics", "Pediatrics", "Radiology",
)


def _asked_departments(question: str) -> set[str]:
    """Departments named explicitly in the question."""
    return {d for d in _KNOWN_DEPARTMENTS if d.lower() in question.lower()}


def _compare(rows: pd.DataFrame, asked: set[str]) -> str:
    """Answer per department instead of collapsing to the majority.

    Used when the question names more than one department. Reports each side on
    its own denominator so the numbers stay comparable, and says plainly when a
    requested department returned nothing.
    """
    scoped = rows[rows["department"].isin(asked)]
    if scoped.empty:
        return ("No retrieved claim belongs to "
                f"{', '.join(sorted(asked))}.")

    parts = [f"Comparing {', '.join(sorted(asked))}: "
             f"{len(scoped)} denials retrieved."]

    for department in sorted(asked):
        subset = scoped[scoped["department"] == department]
        if subset.empty:
            parts.append(f"{department}: no matching claims retrieved.")
            continue
        counts = subset["denial_reason"].value_counts()
        lead, lead_n = counts.index[0], int(counts.iloc[0])
        top = ", ".join(f"{r} {int(c)}"
                        for r, c in counts.head(N_COMPARE_REASONS).items())
        parts.append(
            f"{department}: {len(subset)} denials, most commonly {lead} "
            f"({lead_n} of {len(subset)}, {100 * lead_n / len(subset):.0f}%); "
            f"top reasons {top}."
        )

    ignored = len(rows) - len(scoped)
    if ignored:
        parts.append(f"({ignored} retrieved claim(s) outside "
                     f"{', '.join(sorted(asked))} excluded.)")
    return " ".join(parts)


def generate(question: str, rows: pd.DataFrame) -> str:
    """Answer the question from the retrieved claims.

    Delegates to a model when one is reachable, passing the claims through
    untouched; falls back to the template when none is.
    """
    if rows.empty:
        return "No matching claims found."

    if llm.available():
        try:
            return llm.synthesize(question, rows)
        except RuntimeError as exc:
            print(f"[generator] Claude unavailable ({exc}); using template")

    return render_template(question, rows)


def render_template(question: str, rows: pd.DataFrame) -> str:
    """Deterministic fallback: aggregate the retrieved rows and format them."""
    if rows.empty:
        return "No matching claims found."

    asked = _asked_departments(question)
    if len(asked) > 1:
        return _compare(rows, asked)

    rows, department, dropped = apply_consensus(rows)
    total = len(rows)
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

    cited = _citations(rows, lead_reason)
    if cited:
        parts.append(f"Example claims for {lead_reason}: {', '.join(cited)}.")

    if dropped:
        parts.append(f"({dropped} retrieved claim(s) outside {department} excluded.)")

    warning = _off_target(question, department, rows)
    if warning:
        parts.append(warning)

    return " ".join(parts)
