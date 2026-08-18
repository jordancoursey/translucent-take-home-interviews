## Running this

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # pipeline + evals
pip install -r requirements-dev.txt      # optional: dashboard only
```

First run downloads `all-MiniLM-L6-v2` (~90MB) and caches embeddings to
`data/.embedding_cache/`. Pass `--backend tfidf` anywhere to skip the download.

Answers are synthesised by Claude when a credential is present, and by the
deterministic template otherwise — see `.env.example`. No key is required; the
template is what produces the 5/5 below.

```bash
cp .env.example .env    # optional: add ANTHROPIC_API_KEY to use Claude
```

| Command | What it does | What you get |
|---|---|---|
| `python eval.py` | The provided eval, extended to score both systems on the identical rule | Side-by-side table: baseline **3/5**, pipeline **5/5** |
| `python eval.py --verbose` | Same, plus every answer | Each system's text per question, with the keywords it missed |
| `python main.py ask -q "..."` | Ask a question | One synthesised answer; also appends a trace |
| `python main.py eval-all` | Both evaluators | Answer scores w/ precision, plus retrieval P/R/F1/AUC and the threshold sweep |
| `python -m evaluation.track_results --note "..."` | Record a run | Appends 5 rows to `results/metrics.jsonl`; prints per-question table + run history |
| `python -m evaluation.cheated_eval` | Demonstrate the eval is gameable | A constant string scores 5/5 vs `baseline_agent`'s 3/5 |
| `streamlit run reporting/dashboard.py` | Browse tracked runs | Per-question and per-run charts, answers, tables |

### What each piece is for

| Path | Role |
|---|---|
| `main.py` | Entry point. `ask` + the eval subcommands |
| `retrieval/` | `embedding.py` (query + corpus vectors), `data_prep.py` (rows → documents, cached), `retriever.py` (one similarity step, relative threshold) |
| `generator/` | Retrieved rows → answer text. `llm.py` sends the claims to Claude; `generator.py` falls back to the deterministic template, whose counts are exact by construction |
| `evaluation/` | `cases.py` (the question set), `answer_eval.py`, `retrieval_eval.py`, `cheated_eval.py`, `track_results.py` |
| `observability/` | Append-only trace log at `observability/traces/traces.csv` |
| `reporting/` | Streamlit dashboard over `results/metrics.jsonl` |
| `eval.py`, `baseline_agent.py` | Kept at the root. `baseline_agent` is unchanged apart from the `service_date` fix required to make it run, and serves as the control condition |

`eval.py` was extended to run the baseline and the new pipeline side by side. The
five tests and the `all(keyword in answer)` scoring rule are byte-identical to the
provided version, so both numbers are earned under the original rule; the pipeline
is loaded in a `try` so the file still runs where the model cannot be downloaded.

Modules under `evaluation/` import `main`, so run them as `python -m evaluation.x`
rather than by file path.