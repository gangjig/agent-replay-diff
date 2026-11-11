"""Harness-termination detection and its effect on comparison.

A run whose final step is a harness failure (content-filter crash,
timeout, provider error) did not produce its final answer itself, so it
must be *marked*; with exclusion on, it is also kept out of the verdict.
"""

from __future__ import annotations

import json

from agent_replay_diff.cli import main
from agent_replay_diff.differ import VERDICT_DIVERGENT, VERDICT_STABLE, compare_traces
from agent_replay_diff.termination import detect_harness_termination
from conftest import step


def good_run():
    return [
        step(type="tool_call", tool="search", input={"q": "x"}),
        step(type="final", content="the answer"),
    ]


def crashed_run(error_step):
    return [
        step(type="tool_call", tool="search", input={"q": "x"}),
        error_step,
    ]


# --- detection ---------------------------------------------------------


def test_detects_explicit_marker():
    trace = crashed_run(step(type="final", content="", harness_terminated=True))
    found = detect_harness_termination(trace)
    assert found is not None
    assert found[0] == 1
    assert "harness_terminated" in found[1]


def test_detects_terminated_by_harness():
    trace = crashed_run(step(type="final", content="", terminated_by="harness"))
    assert detect_harness_termination(trace) is not None


def test_detects_content_filter_step_type():
    trace = crashed_run(step(type="content_filter"))
    found = detect_harness_termination(trace)
    assert found is not None
    assert "content_filter" in found[1]


def test_detects_timeout_error_text():
    trace = crashed_run(step(type="error", error="request timed out after 120s"))
    found = detect_harness_termination(trace)
    assert found is not None
    assert "timeout" in found[1]


def test_detects_error_dict_message():
    trace = crashed_run(
        step(type="error", error={"message": "Content filter rejected the prompt"})
    )
    found = detect_harness_termination(trace)
    assert found is not None
    assert "content filter" in found[1]


def test_detects_rate_limit_error_type_field():
    trace = crashed_run(step(type="error", error_type="rate_limit_error"))
    assert detect_harness_termination(trace) is not None


def test_normal_final_answer_not_flagged():
    assert detect_harness_termination(good_run()) is None


def test_mid_trace_error_not_flagged():
    trace = [
        step(type="error", error="request timed out"),
        step(type="tool_call", tool="search", input={"q": "x"}),
        step(type="final", content="recovered answer"),
    ]
    assert detect_harness_termination(trace) is None


def test_agent_side_final_error_not_flagged():
    # A final error naming no harness cause may be the agent's own
    # failure; flagging it would hide a real divergence.
    trace = crashed_run(step(type="error", error="tool failed: file not found"))
    assert detect_harness_termination(trace) is None


def test_empty_trace_not_flagged():
    assert detect_harness_termination([]) is None


# --- comparison integration --------------------------------------------


def test_detected_but_included_by_default():
    report = compare_traces(
        [good_run(), crashed_run(step(type="content_filter"))],
        labels=["good", "crashed"],
    )
    assert report.verdict == VERDICT_DIVERGENT
    assert len(report.harness_terminations) == 1
    term = report.harness_terminations[0]
    assert term.label == "crashed"
    assert term.excluded is False
    as_dict = report.to_dict()
    assert as_dict["harness_terminated"][0]["run"] == "crashed"
    assert as_dict["harness_terminated"][0]["excluded"] is False


def test_excluded_run_cannot_pose_as_divergence():
    report = compare_traces(
        [good_run(), good_run(), crashed_run(step(type="content_filter"))],
        labels=["a", "b", "crashed"],
        exclude_harness_terminated=True,
    )
    assert report.verdict == VERDICT_STABLE
    assert report.labels == ["a", "b"]
    assert len(report.harness_terminations) == 1
    assert report.harness_terminations[0].excluded is True
    assert report.termination_note is not None
    assert "excluded" in report.termination_note


def test_exclusion_falls_back_when_too_few_runs_remain():
    report = compare_traces(
        [good_run(), crashed_run(step(type="content_filter"))],
        labels=["good", "crashed"],
        exclude_harness_terminated=True,
    )
    # Only one agent-produced run remains: nothing to compare against,
    # so the crashed run stays in and the note explains why.
    assert report.labels == ["good", "crashed"]
    assert report.harness_terminations[0].excluded is False
    assert report.termination_note is not None
    assert "fewer than two" in report.termination_note


def test_no_terminations_no_note():
    report = compare_traces([good_run(), good_run()], labels=["a", "b"])
    assert report.harness_terminations == []
    assert report.termination_note is None
    assert report.to_dict()["harness_terminated"] == []


# --- CLI -----------------------------------------------------------------


def test_cli_marks_terminated_run(write_trace, capsys):
    a = write_trace("a.jsonl", good_run())
    b = write_trace("b.jsonl", crashed_run(step(type="content_filter")))
    assert main([str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "harness-terminated runs" in out
    assert "included in comparison" in out


def test_cli_exclude_flag(write_trace, capsys):
    a = write_trace("a.jsonl", good_run())
    b = write_trace("b.jsonl", good_run())
    c = write_trace("c.jsonl", crashed_run(step(type="content_filter")))
    assert main([str(a), str(b), str(c), "--exclude-harness-terminated"]) == 0
    out = capsys.readouterr().out
    assert "excluded from comparison" in out


def test_cli_json_carries_terminations(write_trace, capsys):
    a = write_trace("a.jsonl", good_run())
    b = write_trace("b.jsonl", crashed_run(step(type="content_filter")))
    main([str(a), str(b), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["harness_terminated"][0]["run"] == str(b)
    assert payload["harness_terminated"][0]["step"] == 1
