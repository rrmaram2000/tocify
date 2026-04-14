# Architecture Decisions

Living log of notable technical decisions for this project. Append new entries;
don't rewrite old ones. When a decision is revisited, mark the old entry
**Superseded by ADR-00N** and add a new entry.

---

## ADR-001: Triage papers in batches of 50 rather than one mega-call

**Status:** Accepted **Date:** 2026-04-14 (documented retrospectively)

### Context

After the local keyword prefilter (`keyword_prefilter`) runs, up to
`PREFILTER_KEEP_TOP = 200` candidate papers reach the OpenAI triage stage. Each
paper carries a title, source, published date, and up to
`SUMMARY_MAX_CHARS = 500` characters of RSS summary. The triage step must score
every paper against the user's keywords and narrative and return strict-JSON
ranked output.

The natural question is how to hand those 200 items to the model: one large
call, or several smaller ones. `digest.py:230` (`triage_in_batches`) picks the
latter, defaulting to `BATCH_SIZE = 50` — typically four sequential API calls
per run, followed by a max-score dedup and a final sort.

### Decision

Send papers to the LLM in batches of 50. Run each batch as an independent
`responses.create` call with its own retry loop (`call_openai_triage`, 6
retries, exponential backoff). Merge results by taking the maximum score per
paper ID and sorting globally.

### Alternatives considered

1. **Single mega-call with all ~200 items.** Rejected. Long item lists suffer
   from the lost-in-the-middle phenomenon — items near the middle of the
   sequence receive systematically less attention than items at the start or
   end, producing uneven scoring. The total attention budget also has to be
   spread thinner across 200 items, degrading the quality of each individual
   `why` justification. A single failure (timeout, rate limit, schema reject)
   also wastes the entire run's compute.
2. **Per-item calls (batch size 1).** Rejected. Destroys all ranking context —
   the model can only score each paper in isolation, losing the relative
   calibration that comes from seeing a batch together. Cost and latency scale
   linearly with item count (~200 round trips), and per-call overhead dominates.
3. **Hierarchical triage: rank within batches, then re-rank the survivors.**
   Rejected for now as added complexity without clear benefit at current
   volumes. Revisit if weekly item counts grow substantially or if cross-batch
   calibration drift becomes observably bad.

### Consequences

**Positive**

- Each call fits a context window the model can attend to uniformly, mitigating
  lost-in-the-middle.
- Per-call cost is bounded and predictable. A failed batch retries in isolation;
  other batches' work is preserved.
- Simpler mental model for debugging: batch boundaries align with log lines
  (`Triage batch N/M`), so flaky runs are easy to localize.

**Negative**

- The model never sees all papers at once, so scores across batches are only
  loosely calibrated — a 0.7 in batch 2 is not strictly comparable to a 0.7 in
  batch 4. Final ranking treats them as comparable, which is an acknowledged
  approximation.
- 4× API round trips per run instead of 1, with correspondingly more surface
  area for transient network failures (mitigated by per-batch retries).
- Max-score dedup assumes duplicate IDs across batches are rare (items are
  partitioned, not overlapping); it exists mainly as a safety net for papers
  syndicated through multiple feeds.

### Tunables

- `BATCH_SIZE` (env) — default 50. Lower for even tighter attention budgets;
  higher to reduce round trips at the cost of scoring uniformity.
- `PREFILTER_KEEP_TOP` (env) — default 200. Controls how many items reach this
  stage at all; interacts with `BATCH_SIZE` to determine call count.
