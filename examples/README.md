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
