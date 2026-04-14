# CLAUDE.md — tocify

## What this is

Weekly research-paper triage pipeline. A GitHub Action (or manual run) fetches
new items from journal/preprint RSS feeds, keyword-prefilters them, asks OpenAI
to score each against the user's interests, renders a ranked markdown digest,
and commits it back to the repo as `digest.md`.

## Tech stack

- Python **3.11** in CI (runs fine on 3.12 locally)
- `requirements.txt`: `openai`, `feedparser`, `python-dateutil`
- `httpx` (transitive) used explicitly to build a custom HTTP client
- `pytest` for tests (local dev only, not pinned)

## Project structure

- `digest.py` — single-file pipeline: fetch → prefilter → triage → render
- `feeds.txt` — RSS feeds; supports `Name | URL` and `#` comments
- `interests.md` — keywords and narrative seed
- `prompt.txt` — triage prompt template with `{{KEYWORDS}}`, `{{NARRATIVE}}`,
  `{{ITEMS}}` placeholders
- `digest.md` — generated output, overwritten each run
- `.github/workflows/weekly-digest.yml` — cron + manual trigger
- `tests/` — pytest characterization tests for pure helpers

## Conventions

- **All tunables are env vars** with defaults at the top of `digest.py`
  (`MODEL`, `LOOKBACK_DAYS`, `PREFILTER_KEEP_TOP`, `BATCH_SIZE`,
  `MIN_SCORE_READ`, …). Override in the workflow `env:` block, not in code.
- **Secrets**: `OPENAI_API_KEY` only, stored as a GitHub Actions secret. Never
  commit. `make_openai_client` enforces the `sk-` prefix.
- **Trigger**: cron Mondays 16:00 UTC, or `workflow_dispatch` from the Actions
  tab. Output is committed by the `toc-digest-bot` identity.
- **Proxies deliberately cleared** in the workflow (`HTTP_PROXY=""`, etc.) —
  don't add proxy env vars unless you know why.

## Gotchas

- `interests.md` requires `## Keywords` and `## Narrative` H2 headings; the
  parser is regex-based (`digest.py:97`). Renaming either section breaks
  silently and yields empty keywords/narrative.
- `keyword_prefilter` has a **fallback branch**: if fewer than
  `min(50, keep_top)` items match any keyword, it returns `items[:keep_top]`
  unfiltered — non-matchers can reach the model on quiet weeks.
- `prompt.txt` must contain all three placeholders (`{{KEYWORDS}}`,
  `{{NARRATIVE}}`, `{{ITEMS}}`). Replacement is plain string-substitution —
  missing placeholders fail silently.
- OpenAI call uses strict JSON schema + 6 retries with exponential backoff;
  model output must match `SCHEMA` in `digest.py` or the whole run fails.
- The workflow commits back to `main`. After each scheduled run, local `main`
  diverges from origin — always `git pull` before starting new work.
