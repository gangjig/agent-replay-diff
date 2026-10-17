"""Tests for trace loading: JSONL and JSON-array formats, plus bad inputs."""

from __future__ import annotations

import pytest

from agent_replay_diff.loader import TraceError, load_trace

STEPS = [
    {"step": 0, "type": "message", "content": "hi"},
    {"step": 1, "type": "tool_call", "tool": "search", "input": {"q": "x"}},
]


def test_load_jsonl(write_trace):
    path = write_trace("a.jsonl", STEPS)
    assert load_trace(path) == STEPS


def test_load_json_array(write_trace):
    path = write_trace("a.json", STEPS, fmt="json")
    assert load_trace(path) == STEPS


def test_jsonl_blank_lines_skipped(write_trace):
    path = write_trace(
        "a.jsonl", '{"step": 0}\n\n  \n{"step": 1}\n', fmt="raw"
    )
    assert load_trace(path) == [{"step": 0}, {"step": 1}]


def test_unknown_fields_preserved(write_trace):
    steps = [{"step": 0, "custom_thing": {"nested": [1, 2]}, "vendor": "x"}]
    path = write_trace("a.jsonl", steps)
    assert load_trace(path) == steps


def test_steps_may_omit_fields(write_trace):
    path = write_trace("a.jsonl", [{}, {"type": "message"}])
    assert load_trace(path) == [{}, {"type": "message"}]


def test_unicode_preserved(write_trace):
    steps = [{"content": "Arlington 的人口 — café"}]
    path = write_trace("a.jsonl", steps)
    assert load_trace(path) == steps


def test_missing_file_raises(tmp_path):
    with pytest.raises(TraceError):
        load_trace(tmp_path / "nope.jsonl")


def test_empty_file_raises(write_trace):
    path = write_trace("empty.jsonl", "", fmt="raw")
    with pytest.raises(TraceError):
        load_trace(path)


def test_whitespace_only_file_raises(write_trace):
    path = write_trace("ws.jsonl", "   \n\n", fmt="raw")
    with pytest.raises(TraceError):
        load_trace(path)


def test_empty_array_raises(write_trace):
    path = write_trace("empty.json", [], fmt="json")
    with pytest.raises(TraceError):
        load_trace(path)


def test_malformed_jsonl_reports_line_number(write_trace):
    path = write_trace(
        "bad.jsonl", '{"step": 0}\n{"step": 1}\nnot json\n', fmt="raw"
    )
    with pytest.raises(TraceError) as excinfo:
        load_trace(path)
    assert "line 3" in str(excinfo.value)


def test_malformed_json_array_raises(write_trace):
    path = write_trace("bad.json", '[{"step": 0},', fmt="raw")
    with pytest.raises(TraceError):
        load_trace(path)


def test_non_object_line_raises(write_trace):
    path = write_trace("num.jsonl", '{"step": 0}\n42\n', fmt="raw")
    with pytest.raises(TraceError) as excinfo:
        load_trace(path)
    assert "line 2" in str(excinfo.value)


def test_non_object_array_element_raises(write_trace):
    path = write_trace("bad.json", '[{"step": 0}, "oops"]', fmt="raw")
    with pytest.raises(TraceError) as excinfo:
        load_trace(path)
    assert "element 1" in str(excinfo.value)


def test_single_bare_object_is_a_one_step_trace(write_trace):
    # A lone JSON object doesn't start with "[", so it is read as a
    # one-line JSONL trace — deliberate leniency for single-step runs.
    path = write_trace("obj.json", '{"step": 0}', fmt="raw")
    assert load_trace(path) == [{"step": 0}]


def test_top_level_scalar_json_raises(write_trace):
    path = write_trace("scalar.json", '"just a string"', fmt="raw")
    with pytest.raises(TraceError):
        load_trace(path)
