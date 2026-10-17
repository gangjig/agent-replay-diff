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

The JSON report carries the same information as structured data: per-run step counts, per-step classifications, classification totals, the verdict, the first divergence index, and the raw diffs.

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

## Development

```bash
python3 -m pytest -q
bash examples/run_demo.sh
```

The test suite covers the normalizer pattern by pattern, trace loading (both formats plus malformed inputs), the differ (identical / noise-only / divergence / unequal lengths / 3+ runs), and the CLI end to end, including exit codes.

## License

MIT — see [LICENSE](LICENSE).
