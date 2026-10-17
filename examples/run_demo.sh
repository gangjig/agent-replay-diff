#!/usr/bin/env bash
# Demo: one trace pair that differs only by noise (expected exit 0) and one
# pair that really diverges (expected exit 1). Run from anywhere:
#   bash examples/run_demo.sh
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/../src${PYTHONPATH:+:$PYTHONPATH}"

echo "==================================================================="
echo "Demo 1: noise-only differences (timestamps, UUIDs, latencies,"
echo "        reordered scalar lists) -> expected exit code 0"
echo "==================================================================="
python3 -m agent_replay_diff noise_run_a.jsonl noise_run_b.jsonl
echo "exit code: $? (expected 0)"
echo
echo "==================================================================="
echo "Demo 2: the search returns a different top result at step 2, so"
echo "        the runs fetch different pages -> expected exit code 1"
echo "==================================================================="
python3 -m agent_replay_diff diverge_run_a.jsonl diverge_run_b.jsonl
echo "exit code: $? (expected 1)"
