# agent-replay-diff

**Compare repeated runs of the same agent task: normalize away the volatile noise (timestamps, UUIDs, latencies, ordering) in recorded traces, diff the runs step by step, and pinpoint the first step where the agent actually behaved differently.**

## Why this exists

Agent harnesses are nondeterministic. Run the same task twice and the two traces differ almost everywhere — fresh timestamps, new request IDs, different latencies, tool results in a different order — even when the agent did exactly the same thing. Buried somewhere in that noise there may also be a *real* divergence: the step where a tool returned something different and the run went off in another direction. Eyeballing raw traces makes the two indistinguishable, and naive text diffs drown the one meaningful change in hundreds of meaningless ones.

agent-replay-diff separates the two. It normalizes volatile fields out of each recorded step, classifies every step as **identical**, **noise-only** (differences erased by normalization), or **divergent**, and reports the first divergent step with a readable diff of the raw content — the place debugging should start.

## Credits / motivation

Inspired by the harness-nondeterminism findings reported in Latent Space AINews (2026-10-16): repeated runs of the same agent harness vary substantially, much of the variance comes from un-normalized tool outputs and uncontrolled sampling settings rather than from the model, and normalizing tool outputs removes much of it. This tool turns that observation into a check you can run on your own recorded traces: https://www.latent.space/p/ainews-high-taste-testers-ep-1

## Install

Requires Python 3.11+. Standard library only — no runtime dependencies.

```bash
pip install -e .
```

Or run it straight from a source checkout without installing:

```bash
PYTHONPATH=src python3 -m agent_replay_diff run_a.jsonl run_b.jsonl
```

## Trace format

A trace is the recording of one run: a sequence of steps. Two shapes are supported:

- **JSONL** — one JSON object per line, one step per line.
- **JSON array** — one JSON document containing an array of step objects.

A step is a JSON object, conventionally:

```json
{"step": 0, "type": "tool_call", "tool": "search", "input": {"query": "arlington population"}}
{"step": 1, "type": "tool_result", "tool": "search", "output": {"results": ["..."]}}
{"step": 2, "type": "message", "role": "assistant", "content": "The population is ..."}
```

`type` is typically `tool_call`, `tool_result`, or `message`, but the format is deliberately lenient: steps may omit any field, and unknown fields are preserved and compared like any other. What is rejected: a step that is not a JSON object, malformed JSON, or an empty file (exit code 2).

## Usage

```bash
# Compare two recorded runs
agent-replay-diff run_a.jsonl run_b.jsonl

# Three or more runs at once (each is compared against the first)
agent-replay-diff run_a.jsonl run_b.jsonl run_c.jsonl

# Machine-readable report for CI
agent-replay-diff run_a.jsonl run_b.jsonl --json

# Your traces carry volatile fields the defaults don't know about
agent-replay-diff run_a.jsonl run_b.jsonl --ignore-key seed --ignore-key elapsed_ns

# Volatile content with a custom shape (repeatable regex)
agent-replay-diff run_a.jsonl run_b.jsonl --pattern 'job-[0-9]+'

# Treat scalar-list order as meaningful
agent-replay-diff run_a.jsonl run_b.jsonl --no-sort-lists

# Runs whose harness compacts context mid-run: don't let the
# summarizer's wording pose as a divergence
agent-replay-diff run_a.jsonl run_b.jsonl --compaction-aware

# One run died on the provider's content filter: don't let a
# harness failure pose as agent instability
agent-replay-diff run_a.jsonl run_b.jsonl run_c.jsonl --exclude-harness-terminated

# Longer first-divergence diffs
agent-replay-diff run_a.jsonl run_b.jsonl --max-diff-lines 200
```

Example text report on a pair that diverges at step 2 (see `examples/` for the full walkthrough):

```
agent-replay-diff report
========================
run_a.jsonl: 5 steps
run_b.jsonl: 5 steps
compared steps: 5

identical:   0
noise-only:  2
divergent:   3

verdict: DIVERGENT
first divergence: step 2

--- run_a.jsonl step 2 (raw)
+++ run_b.jsonl step 2 (raw)
@@ -1,17 +1,17 @@
 {
-  "call_id": "1a2b3c4d-...",
...
```

(the diff shows the full raw step, so you see real recorded values rather than placeholders.)

The JSON report carries the same information as structured data: per-run step counts, per-step classifications, classification totals, the verdict, the first divergence index, the raw diffs, and any harness-terminated runs (`harness_terminated`, with `termination_note`).

## How it works

### Normalization

Each step is rewritten recursively (dicts, lists, strings, numbers) into a canonical form:

- **Volatile keys** — values under keys named `timestamp`, `created_at`, `duration_ms`, `latency`, `elapsed`, `request_id`, `trace_id`, etc. (plus *any* key ending in `_at` or `_ms`, plus your `--ignore-key`s) are replaced with `"<volatile>"`. The key stays, so gaining or losing the field is still a structural difference.
- **Volatile strings** — ISO-8601 datetimes, dates, times, UUIDs, long hex strings (32+ chars), and epoch timestamps (10-digit seconds / 13-digit milliseconds) are replaced with placeholders, including when embedded in longer text. Epoch-valued numbers are replaced too.
- **Custom patterns** — your `--pattern` regexes run first; matches become `"<custom>"`.
- **Canonical shape** — dict keys are sorted; lists whose elements are all scalars are sorted (tool results often return the same set in arbitrary order). Lists containing objects are never reordered. `--no-sort-lists` turns scalar sorting off.

### Comparison

Runs are aligned by step index. A step is **identical** if the raw steps are equal in every run, **noise-only** if they differ raw but are equal after normalization, and **divergent** otherwise — including steps missing from some run when the runs have unequal lengths. The first divergent step gets unified-style diffs of the *raw* step content (baseline = first run) so you see the real values, not placeholders.

Verdicts: `stable` (every step identical), `stable_after_normalization` (no real divergence), `divergent`.

### Harness-terminated runs

Some runs never finish: the provider's content filter crashes the call, the request times out, the provider errors out — and the harness writes *something* as the final answer (or nothing). That answer was not produced by the agent, so counting the run as a divergence misreads a harness failure as agent instability. This is not a corner case: in the failure analysis of *Dependable Long-Horizon LLM Agents* ([arXiv:2611.13254](https://arxiv.org/abs/2611.13254)), 45% of human-annotated failures on the Who&When benchmark ended with a provider content-filter crash in which the harness, not the agent, produced the final answer.

Every run's **final step** is checked for harness-failure signals: an explicit `harness_terminated: true` or `terminated_by: "harness"` marker, a final-step type naming a harness failure (`content_filter`, `provider_error`, `timeout`, `harness_error`, ...), or a final-step `error` whose text names a harness cause (content filter, timeout, rate limit, provider/connection failure). Detected runs are always listed in the report — text and JSON (`harness_terminated`) — with the step and the signal that matched. Detection looks at the final step only, on purpose: an error mid-trace is normal agent life (a failed tool call that gets retried), and a bare final `type: "error"` naming no harness cause is *not* flagged, because it may be the agent's own failure and mislabeling it would hide a real divergence.

By default a detected run still takes part in the comparison (the report marks it `included in comparison`). `--exclude-harness-terminated` keeps detected runs out of the step comparison and the verdict entirely — the flakiness-gate setup: a provider outage should not fail your CI as agent instability. Exclusion applies only when at least two non-terminated runs remain to compare; otherwise the runs are all compared and the report's `termination_note` says why.

### Compaction-aware mode

Modern harnesses compact their context mid-run: the earlier steps are replaced by a generated summary and the agent continues from it (OpenAI's Agents SDK does this in the background, on by default). Two runs of the same task then produce *different summaries* — and different wording in every message that quotes the summary afterwards — even when the agent behaved identically. Strict mode reports that as a divergence at the compaction step: the most common fork becomes a fake one.

`--compaction-aware` treats each run's first compaction step (type `compaction` or `context_compaction`; add your harness's name with `--compaction-type`) as an expected divergence boundary and compares in three phases:

1. **Before the boundary: strict.** Steps before the earliest compaction are compared exactly as above — the summary doesn't exist yet, so there is no excuse for drift.
2. **At the boundary: spot-check, not diff.** The summaries are never worded alike, so they are checked instead: lengths within 2× of each other, and the key entities they mention (dotted identifiers like `auth.py`, `snake_case` names, numbers, quoted spans) overlapping with Jaccard similarity ≥ 0.5. A summary that dropped the file under repair, or ballooned to ten times the length, is a real divergence, reported at the boundary step with the summaries diffed.
3. **After the boundary: behavior only.** Message wording may drift (it quotes a differently-worded summary), so the tail is compared on what the agent *did*: the tool-call sequence (tool names with normalized inputs, in order) and the final step. Everything else in the tail is counted as ignored, and the count is in the report.

```
compaction-aware comparison:
  boundary (first compaction): run_a.jsonl step 3, run_b.jsonl step 3
  summaries: lengths 173/160 chars (ratio 1.08), entity overlap 1.00 — OK
  tail: tool calls per run [2, 2], final step equal, 2 post-compaction step(s) ignored

verdict: STABLE AFTER NORMALIZATION (differences were volatile noise only)
```

The same pair in strict mode reports `DIVERGENT, first divergence: step 3` — the compaction step itself (see `examples/compact_run_*.jsonl`, Demo 3 in `examples/README.md`). The mode applies only when **every** run contains a compaction step: if one run never compacted, the runs genuinely took different paths, and the comparison falls back to strict mode with a note in the report. Runs may compact at different steps; a run's steps between the earliest boundary and its own boundary are covered only by its summary spot-check. Only the first compaction per run is a boundary — later re-compactions sit in the tail and are ignored like other non-tool steps.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | The runs are equivalent: every step is identical or differs only by noise that normalization erases |
| 1 | A real divergence was found (see `first divergence` in the report) |
| 2 | Usage or input error: fewer than two files, an unreadable/malformed trace, or an invalid `--pattern` regex |

Exit code 1 makes the tool usable as a flakiness gate: record N runs of a task in CI and fail when they stop agreeing.

## Limitations

agent-replay-diff is honest about what it cannot do:

- **Index-based alignment only.** Steps are matched by position, not by meaning. If one run inserts an extra step early (a retry, an extra tool call), every later step shifts and will report as divergent even if the runs "did the same thing" semantically. There is no semantic step matching or edit-distance realignment.
- **Normalization can mask real differences.** If a genuinely meaningful value happens to look volatile (a real ID that is UUID-shaped, a meaningful timestamp, a number in epoch range), or your `--pattern` / `--ignore-key` choices are too aggressive, real differences will be classified as noise. Start from the defaults and widen them only with evidence.
- **JSON traces only.** Traces must be JSONL or a JSON array of step objects. Other recording formats (message lists in code, vendor-specific event streams) need converting first.
- **Equality is structural, not semantic.** Two steps that say the same thing in different words are divergent; two steps with the same words in a different (object) list order are divergent. The tool finds where runs parted, not whether the parting mattered.
- **Harness-termination detection is heuristic.** It reads the final step's markers, type, and error text. A harness that records a filter crash in some other shape — or silently, with no final step at all — leaves nothing to detect, and a message that merely *mentions* a timeout in its error text will be flagged even if the agent caused it. Widen or narrow the signal set to your harness's conventions before gating CI on exclusion.
- **Compaction-aware mode checks summaries by proxy.** Length ratio and entity overlap catch summaries that dropped the topic or ballooned, but a summary can pass both checks and still misstate a fact the entities don't capture. The spot-check removes the *fake* forks; it does not certify the summaries as faithful.

## Development

```bash
python3 -m pytest -q
bash examples/run_demo.sh
```

The test suite covers the normalizer pattern by pattern, trace loading (both formats plus malformed inputs), the differ (identical / noise-only / divergence / unequal lengths / 3+ runs), and the CLI end to end, including exit codes.

## Build your own

This repository is small enough to regenerate from a prompt. Paste the
following into an LLM and it should produce a closely similar project:

> Write a small Python command-line tool, standard library only, that
> compares repeated recorded runs of the same agent task and pinpoints
> the first step where the agent actually behaved differently. Because
> raw traces differ almost everywhere (fresh timestamps, request ids,
> latencies, tool results arriving in a different order), the tool first
> normalizes volatile fields away, then classifies every step as
> identical, noise-only (different raw, equal after normalization), or
> divergent, and reports the first divergent step with a readable diff
> of the raw content. Its verdicts distinguish fully stable runs, runs
> that differ only by noise, and genuinely divergent ones, with a
> machine-readable report mode for CI. It also has a compaction-aware
> mode for harnesses that summarize their context mid-run: steps before
> the compaction are compared strictly, the summaries themselves are
> spot-checked for length and key entities instead of diffed word by
> word, and the tail is compared on tool-call behavior only.

## License

MIT — see [LICENSE](LICENSE).
