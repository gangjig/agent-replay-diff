# Changelog

## 0.2.0 — 2026-11-01

- **Compaction-aware comparison (`--compaction-aware`).** Harnesses
  that compact context mid-run (OpenAI's Agents SDK now does it in
  the background, on by default) replace earlier steps with a
  generated summary — and two runs' summaries are never worded alike,
  so strict comparison reported the compaction step itself as the
  first divergence of runs that behaved identically. The new mode
  treats each run's first compaction step (type `compaction` or
  `context_compaction`, extensible via `--compaction-type`) as an
  expected divergence boundary: strict comparison before it; at it, a
  spot-check instead of a diff (summary lengths within 2×, key
  entities — dotted identifiers, snake_case names, numbers, quoted
  spans — overlapping with Jaccard ≥ 0.5); after it, behavior only
  (tool-call sequence with normalized inputs, plus the final step),
  with the ignored-step count reported. The mode applies only when
  every run compacted, and falls back to strict comparison with a
  note otherwise. Reports (text and JSON) carry the boundary indices,
  summary check numbers, and tail outcome; the JSON report also gains
  top-level `mode`, `compaction`, and `compaction_note` fields.
  Motivated by OpenAI Python SDK 3.42.0 turning background context
  compaction on by default (via AI Weekly):
  https://aiweekly.co/alerts/openai-python-sdk-3420-adds-agent-context-compaction-controls

## 0.1.0 — 2026-10-24

- Initial release: step-by-step comparison of repeated agent runs
  with normalization of volatile content (timestamps, UUIDs, request
  IDs, latencies, scalar-list order), first-divergence reporting with
  raw unified diffs, text and JSON reports, and a 0/1/2 exit-code
  contract for CI gating.
