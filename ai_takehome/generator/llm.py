"""Claude-backed answer synthesis over retrieved claims.

This is the augmentation step the pipeline previously lacked: retrieved rows are
rendered into prompt context and handed to a model, rather than being collapsed
into counts by a template first.

Each retrieved claim is passed whole -- the same sentence used to build its
embedding -- so nothing is pre-aggregated or filtered on the model's behalf. It
sees the evidence, not a summary of it.

Optional by design. If the SDK or a credential is missing, `available()` returns
False and generator.py falls back to the deterministic template, so eval.py runs
for someone with no API key (which the exercise README requires).

Credentials come from the environment: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
or an `ant auth login` profile. See .env.example.
"""

from __future__ import annotations

import functools
import os

import pandas as pd

from retrieval.data_prep import row_to_document

MODEL = "claude-opus-5"
MAX_TOKENS = 8192

SYSTEM = """You answer questions about health-insurance claim denials for a \
claims-operations analyst.

Rules:
- Use only the claims provided in the user message. Never invent a claim, a \
department, a denial reason, or a number.
- The claims given to you were selected by a similarity search and may include \
rows that do not belong to the question's subject. Ignore any that do not fit, \
and say so if you drop a meaningful number.
- Counts must be exact. Count the provided rows; do not estimate.
- Note the `status` field: a claim marked Paid was ultimately paid, so it is not \
an outstanding denial. Say which basis you are counting on.
- Cite specific claim ids as evidence for the main finding.
- If the provided claims cannot answer the question, say so plainly instead of \
guessing.
- Two or three sentences. No preamble, no restating the question."""


def _credentials_present() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # An `ant auth login` profile is also a valid credential source.
    config = os.path.expanduser("~/.config/anthropic")
    return os.path.isdir(config)


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """True when an Anthropic client can plausibly be constructed."""
    import importlib.util
    if importlib.util.find_spec("anthropic") is None:
        return False
    return _credentials_present()


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def build_context(rows: pd.DataFrame) -> str:
    """Render retrieved claims as prompt context -- the augmentation step.

    Uses the same rendering that produced each row's embedding, so what the model
    reads is what retrieval matched on.
    """
    return "\n".join(f"- {row_to_document(row)}" for _, row in rows.iterrows())


def build_prompt(question: str, rows: pd.DataFrame) -> str:
    """The user-turn content: the question plus every retrieved claim."""
    return (
        f"Question: {question}\n\n"
        f"Retrieved claims ({len(rows)}):\n{build_context(rows)}"
    )


def synthesize(question: str, rows: pd.DataFrame) -> str:
    """Answer from the retrieved claims. Raises on API failure; caller falls back."""
    import anthropic

    try:
        response = _client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": build_prompt(question, rows)}],
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Anthropic credentials rejected") from None
    except anthropic.RateLimitError:
        raise RuntimeError("Anthropic rate limit hit") from None
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error {exc.status_code}") from None
    except anthropic.APIConnectionError:
        raise RuntimeError("Could not reach the Anthropic API") from None

    if response.stop_reason == "refusal":
        raise RuntimeError("Model declined to answer")

    return " ".join(
        block.text.strip() for block in response.content if block.type == "text"
    ).strip()
