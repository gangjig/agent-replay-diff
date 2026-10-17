"""End-to-end CLI tests: exit codes 0/1/2, options, JSON output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_replay_diff.cli import main

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def base_steps():
    return [
        {"step": 0, "type": "message", "content": "do the task"},
        {"step": 1, "type": "tool_call", "tool": "search", "input": {"q": "x"}},
        {"step": 2, "type": "message", "content": "done"},
    ]


def test_exit_0_identical(write_trace, capsys):
    a = write_trace("a.jsonl", base_steps())
    b = write_trace("b.jsonl", base_steps())
    assert main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "STABLE" in out
    assert "first divergence: none" in out


def test_exit_0_noise_only(write_trace, capsys):
    a_steps = base_steps()
    b_steps = base_steps()
    a_steps[1]["timestamp"] = "2026-10-17T08:00:05Z"
    b_steps[1]["timestamp"] = "2026-10-17T09:00:05Z"
    a_steps[1]["duration_ms"] = 10
    b_steps[1]["duration_ms"] = 999
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b)]) == 0
    assert "STABLE AFTER NORMALIZATION" in capsys.readouterr().out


def test_exit_1_divergent_reports_first_step(write_trace, capsys):
    a_steps = base_steps()
    b_steps = base_steps()
    b_steps[2] = {"step": 2, "type": "message", "content": "different ending"}
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "DIVERGENT" in out
    assert "first divergence: step 2" in out
    assert "different ending" in out  # raw diff shows the real values


def test_exit_2_single_file():
    with pytest.raises(SystemExit) as excinfo:
        main(["only-one.jsonl"])
    assert excinfo.value.code == 2


def test_exit_2_missing_file(write_trace, capsys):
    a = write_trace("a.jsonl", base_steps())
    assert main([str(a), "does-not-exist.jsonl"]) == 2
    assert "error" in capsys.readouterr().err


def test_exit_2_malformed_file(write_trace):
    a = write_trace("a.jsonl", base_steps())
    bad = write_trace("bad.jsonl", "{not json\n", fmt="raw")
    assert main([str(a), str(bad)]) == 2


def test_exit_2_invalid_pattern(write_trace):
    a = write_trace("a.jsonl", base_steps())
    b = write_trace("b.jsonl", base_steps())
    assert main([str(a), str(b), "--pattern", "[unclosed"]) == 2


def test_exit_2_bad_max_diff_lines(write_trace):
    a = write_trace("a.jsonl", base_steps())
    b = write_trace("b.jsonl", base_steps())
    with pytest.raises(SystemExit) as excinfo:
        main([str(a), str(b), "--max-diff-lines", "0"])
    assert excinfo.value.code == 2


def test_json_output_parses(write_trace, capsys):
    a_steps = base_steps()
    b_steps = base_steps()
    b_steps[0]["timestamp"] = "2026-10-17T08:00:05Z"
    a_steps[0]["timestamp"] = "2026-10-17T07:00:05Z"
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "stable_after_normalization"
    assert data["counts"] == {"identical": 2, "noise_only": 1, "divergent": 0}
    assert data["first_divergence"] is None
    assert len(data["steps"]) == 3


def test_json_output_divergent(write_trace, capsys):
    b_steps = base_steps()
    b_steps[1]["input"] = {"q": "other"}
    a = write_trace("a.jsonl", base_steps())
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "divergent"
    assert data["first_divergence"] == 1
    assert data["divergence_diffs"]


def test_ignore_key_option(write_trace):
    a_steps = base_steps()
    b_steps = base_steps()
    a_steps[1]["seed"] = 111
    b_steps[1]["seed"] = 222
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b)]) == 1  # divergent without the option
    assert main([str(a), str(b), "--ignore-key", "seed"]) == 0


def test_pattern_option(write_trace):
    a_steps = base_steps()
    b_steps = base_steps()
    a_steps[1]["output_ref"] = "job-1001"
    b_steps[1]["output_ref"] = "job-2002"
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b)]) == 1
    assert main([str(a), str(b), "--pattern", r"job-\d+"]) == 0


def test_no_sort_lists_option(write_trace):
    a_steps = base_steps()
    b_steps = base_steps()
    a_steps[1]["input"] = {"tags": ["a", "b"]}
    b_steps[1]["input"] = {"tags": ["b", "a"]}
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b)]) == 0  # order normalized away by default
    assert main([str(a), str(b), "--no-sort-lists"]) == 1


def test_max_diff_lines_option(write_trace, capsys):
    a_steps = [{"output": {f"k{i}": i for i in range(40)}}]
    b_steps = [{"output": {f"k{i}": -i for i in range(40)}}]
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    assert main([str(a), str(b), "--max-diff-lines", "5"]) == 1
    assert "diff truncated" in capsys.readouterr().out


def test_three_files(write_trace, capsys):
    a = write_trace("a.jsonl", base_steps())
    b = write_trace("b.jsonl", base_steps())
    c_steps = base_steps()
    c_steps[0] = {"step": 0, "type": "message", "content": "other task"}
    c = write_trace("c.jsonl", c_steps)
    assert main([str(a), str(b), str(c)]) == 1
    assert "first divergence: step 0" in capsys.readouterr().out


def test_json_array_input_files(write_trace):
    a = write_trace("a.json", base_steps(), fmt="json")
    b = write_trace("b.json", base_steps(), fmt="json")
    assert main([str(a), str(b)]) == 0


def test_python_module_invocation(write_trace):
    a = write_trace("a.jsonl", base_steps())
    b_steps = base_steps()
    b_steps[2] = {"step": 2, "type": "message", "content": "nope"}
    b = write_trace("b.jsonl", b_steps)
    env = dict(os.environ, PYTHONPATH=str(SRC_DIR))
    proc = subprocess.run(
        [sys.executable, "-m", "agent_replay_diff", str(a), str(b)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    assert "first divergence: step 2" in proc.stdout


def test_python_module_invocation_missing_file(write_trace):
    a = write_trace("a.jsonl", base_steps())
    env = dict(os.environ, PYTHONPATH=str(SRC_DIR))
    proc = subprocess.run(
        [sys.executable, "-m", "agent_replay_diff", str(a), "ghost.jsonl"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 2
