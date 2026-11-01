# Examples

Two recorded trace pairs of small agent tasks, showing the two outcomes
agent-replay-diff exists to separate. Run the whole demo with:

```bash
bash examples/run_demo.sh
```

(or `pip install -e .` once at the repo root and call `agent-replay-diff`
directly on the files below).

## Demo 1 — noise only (`noise_run_a.jsonl` vs `noise_run_b.jsonl`)

The same weather task recorded twice. Every step differs on the surface —
epoch `ts` values, ISO timestamps, call UUIDs, `request_id`, `duration_ms` /
`latency_ms`, and the `conditions` list comes back in a different order — but
the agent did exactly the same thing:

```bash
agent-replay-diff examples/noise_run_a.jsonl examples/noise_run_b.jsonl
```

```
agent-replay-diff report
========================
examples/noise_run_a.jsonl: 4 steps
examples/noise_run_b.jsonl: 4 steps
compared steps: 4

identical:   0
noise-only:  4
divergent:   0

verdict: STABLE AFTER NORMALIZATION (differences were volatile noise only)
first divergence: none
```

Exit code **0**: the runs are equivalent.

## Demo 2 — a real divergence (`diverge_run_a.jsonl` vs `diverge_run_b.jsonl`)

The same research task recorded twice. Steps 0–1 differ only by noise, but at
step 2 the search tool returns a **different top result** (Census Reporter in
run A, Wikipedia in run B), so the runs fetch different pages at step 3 and
word the final answer differently at step 4:

```bash
agent-replay-diff examples/diverge_run_a.jsonl examples/diverge_run_b.jsonl
```

```
agent-replay-diff report
========================
examples/diverge_run_a.jsonl: 5 steps
examples/diverge_run_b.jsonl: 5 steps
compared steps: 5

identical:   0
noise-only:  2
divergent:   3

verdict: DIVERGENT
first divergence: step 2

--- examples/diverge_run_a.jsonl step 2 (raw)
+++ examples/diverge_run_b.jsonl step 2 (raw)
@@ -1,17 +1,17 @@
 {
-  "call_id": "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
-  "latency_ms": 342,
+  "call_id": "7e8f9a0b-1c2d-4e3f-a4b5-c6d7e8f9a0b1",
+  "latency_ms": 518,
   "output": {
     "results": [
       {
-        "snippet": "Arlington County, VA has a population of 238,643.",
-        "title": "Census Reporter — Arlington County, VA",
-        "url": "https://censusreporter.org/profiles/05000US51013/"
+        "snippet": "The population was 238,643 at the 2024 estimate.",
+        "title": "Arlington County, Virginia — Wikipedia",
+        "url": "https://en.wikipedia.org/wiki/Arlington_County,_Virginia"
       }
     ]
```

Note the diff intentionally shows the **raw** step — including the noise
fields, so you see the real recorded values — while the classification that
picked this step was computed on the normalized form.

Exit code **1**: step 2 is where the runs actually parted ways, which is
where debugging should start — everything after it differs *because* of
step 2, not independently.

## Demo 3 — compaction (`compact_run_a.jsonl` vs `compact_run_b.jsonl`)

The same bug-fix task recorded twice. Both runs compact their context at
step 3 — but the two summaries are worded completely differently, and the
step-4 message quotes each run's own summary, so it differs too. The agent
*did* the same thing in both runs: same reads, same edit, same tests, same
final message.

Strict mode falls for it:

```bash
agent-replay-diff examples/compact_run_a.jsonl examples/compact_run_b.jsonl
# verdict: DIVERGENT, first divergence: step 3 (the compaction step itself)
```

Compaction-aware mode treats step 3 as the expected boundary — strict
before it, a spot-check of the summaries at it (lengths within 2×, key
entities overlapping), and behavior-only comparison after it:

```bash
agent-replay-diff examples/compact_run_a.jsonl examples/compact_run_b.jsonl --compaction-aware
```

```
agent-replay-diff report
========================
examples/compact_run_a.jsonl: 8 steps
examples/compact_run_b.jsonl: 8 steps
compared steps: 3

identical:   2
noise-only:  1
divergent:   0

compaction-aware comparison:
  boundary (first compaction): examples/compact_run_a.jsonl step 3, examples/compact_run_b.jsonl step 3
  summaries: lengths 173/160 chars (ratio 1.08), entity overlap 1.00 — OK
  tail: tool calls per run [2, 2], final step equal, 2 post-compaction step(s) ignored

verdict: STABLE AFTER NORMALIZATION (differences were volatile noise only)
first divergence: none
```

Exit code **0** with `--compaction-aware`, **1** without: the compaction
step is where the runs' *records* part, not where the runs themselves do.
