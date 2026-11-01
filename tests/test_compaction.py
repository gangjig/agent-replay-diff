"""Compaction-aware comparison: the boundary, the spot-check, the tail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_replay_diff.cli import main
from agent_replay_diff.compaction import (
    compare_traces_compaction_aware,
    extract_entities,
    first_compaction_index,
    is_compaction_step,
    summary_text,
)
from agent_replay_diff.differ import compare_traces

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SUMMARY_A = (
    "Task: fix the auth bug in auth.py. Found: validate_token compares "
    "expiry with the client clock. Next: change validate_token to use "
    "the server clock, then run the auth tests."
)
SUMMARY_B = (
    "Fixing the auth bug in auth.py: validate_token trusted the client "
    "clock for expiry. Plan: switch validate_token to the server clock "
    "and run the auth test suite."
)


def msg(content, **extra):
    return {"type": "message", "content": content, **extra}


def tool(name, inp, **extra):
    return {"type": "tool_call", "tool": name, "input": inp, **extra}


def compact(summary, **extra):
    return {"type": "compaction", "summary": summary, **extra}


def run_a():
    return [
        msg("Fix the auth bug in auth.py", role="user"),
        tool("read_file", {"path": "auth.py"}, duration_ms=120),
        msg("The token check compares the expiry against the client clock."),
        compact(SUMMARY_A),
        msg("Now I'll patch the validation function."),
        tool("edit_file", {"path": "auth.py", "change": "server clock"}),
        tool("run_tests", {"suite": "auth"}),
        msg("Done: auth bug fixed, 12 tests pass."),
    ]


def run_b():
    return [
        msg("Fix the auth bug in auth.py", role="user"),
        tool("read_file", {"path": "auth.py"}, duration_ms=940),
        msg("The token check compares the expiry against the client clock."),
        compact(SUMMARY_B),
        msg("Patching the token validation now."),
        tool("edit_file", {"path": "auth.py", "change": "server clock"}),
        tool("run_tests", {"suite": "auth"}),
        msg("Done: auth bug fixed, 12 tests pass."),
    ]


def test_detection_and_summary_text():
    assert is_compaction_step({"type": "compaction"})
    assert is_compaction_step({"type": "Context_Compaction"})
    assert not is_compaction_step({"type": "message"})
    assert not is_compaction_step({})
    assert first_compaction_index(run_a()) == 3
    assert first_compaction_index([msg("x")]) is None
    assert summary_text({"type": "compaction", "summary": "s"}) == "s"
    assert summary_text({"type": "compaction", "text": "t"}) == "t"
    assert summary_text({"type": "compaction"}) == ""
    assert is_compaction_step({"type": "squash"}, types=("squash",))


def test_extract_entities():
    entities = extract_entities(SUMMARY_A)
    assert "auth.py" in entities
    assert "validate_token" in entities
    assert extract_entities("no entities here") == set()


def test_happy_path_is_equivalent_not_a_fork():
    report = compare_traces_compaction_aware([run_a(), run_b()])
    assert report.mode == "compaction_aware"
    assert report.verdict == "stable_after_normalization"
    assert report.exit_code == 0
    assert report.first_divergence is None
    c = report.compaction
    assert c is not None
    assert c.boundary_indices == [3, 3]
    assert c.summary_ok is True
    assert c.summary_entity_overlap == 1.0
    assert c.tail_tool_calls == [2, 2]
    assert c.tail_tool_mismatch is None
    assert c.final_equal is True
    assert c.ignored_steps == 2  # each run's reworded step-4 message
    data = report.to_dict()
    assert data["mode"] == "compaction_aware"
    assert data["compaction"]["summary_ok"] is True


def test_strict_mode_flags_the_same_pair_as_divergent():
    report = compare_traces([run_a(), run_b()])
    assert report.verdict == "divergent"
    assert report.first_divergence == 3  # the compaction step itself


def test_prefix_divergence_still_caught_strictly():
    a, b = run_a(), run_b()
    b[2] = msg("Something entirely different before compaction.")
    report = compare_traces_compaction_aware([a, b])
    assert report.verdict == "divergent"
    assert report.first_divergence == 2
    assert report.exit_code == 1


def test_summary_entity_mismatch_is_a_boundary_divergence():
    a, b = run_a(), run_b()
    b[3] = compact(
        "Task: fix the billing bug in billing.py. Found: charge_card "
        "rounds totals down. Next: change charge_card rounding."
    )
    report = compare_traces_compaction_aware([a, b])
    assert report.verdict == "divergent"
    assert report.first_divergence == 3
    assert report.compaction.summary_ok is False
    assert report.compaction.summary_entity_overlap == 0.0
    assert report.divergence_diffs  # the summaries are shown


def test_summary_length_blowup_is_a_boundary_divergence():
    a, b = run_a(), run_b()
    b[3] = compact("auth.py " * 200)  # same entity, ~10x the length
    report = compare_traces_compaction_aware([a, b])
    assert report.verdict == "divergent"
    assert report.compaction.summary_ok is False
    assert report.compaction.summary_length_ratio > 2.0


def test_tail_tool_mismatch_is_divergent():
    a, b = run_a(), run_b()
    b[6] = tool("run_tests", {"suite": "billing"})
    report = compare_traces_compaction_aware([a, b])
    assert report.verdict == "divergent"
    assert report.first_divergence == 3  # reported at the boundary
    assert "run_tests" in report.compaction.tail_tool_mismatch


def test_tail_final_step_mismatch_is_divergent():
    a, b = run_a(), run_b()
    b[7] = msg("Done: auth bug fixed, 9 tests pass.")
    report = compare_traces_compaction_aware([a, b])
    assert report.verdict == "divergent"
    assert report.compaction.final_equal is False


def test_tail_wording_drift_is_ignored():
    a, b = run_a(), run_b()
    b[4] = msg("Completely reworded continuation, same behavior.")
    report = compare_traces_compaction_aware([a, b])
    assert report.verdict == "stable_after_normalization"
    assert report.exit_code == 0


def test_offset_boundaries():
    # B compacts two steps later; its extra pre-compaction steps are
    # covered only by its summary spot-check.
    a = run_a()
    b = [
        msg("Fix the auth bug in auth.py", role="user"),
        tool("read_file", {"path": "auth.py"}, duration_ms=940),
        msg("The token check compares the expiry against the client clock."),
        msg("Let me also check the session module."),
        tool("read_file", {"path": "session.py"}),
        compact(SUMMARY_B),
        tool("edit_file", {"path": "auth.py", "change": "server clock"}),
        tool("run_tests", {"suite": "auth"}),
        msg("Done: auth bug fixed, 12 tests pass."),
    ]
    report = compare_traces_compaction_aware([a, b])
    assert report.compaction.boundary_indices == [3, 5]
    assert report.verdict == "stable_after_normalization"
    assert report.exit_code == 0


def test_fallback_when_a_run_never_compacts():
    a, b = run_a(), [step for step in run_b() if step["type"] != "compaction"]
    report = compare_traces_compaction_aware([a, b])
    assert report.mode == "strict"
    assert report.compaction is None
    assert "not applied" in report.compaction_note
    assert report.verdict == "divergent"  # strict comparison of drifted runs


def test_cli_compaction_aware_on_examples(capsys):
    a = EXAMPLES / "compact_run_a.jsonl"
    b = EXAMPLES / "compact_run_b.jsonl"
    assert main([str(a), str(b), "--compaction-aware"]) == 0
    out = capsys.readouterr().out
    assert "compaction-aware comparison:" in out
    assert "summaries:" in out and "OK" in out
    # And the same pair without the flag is a divergence.
    assert main([str(a), str(b)]) == 1
    assert "first divergence: step 3" in capsys.readouterr().out


def test_cli_compaction_json(write_trace, capsys):
    a = write_trace("a.jsonl", run_a())
    b = write_trace("b.jsonl", run_b())
    assert main([str(a), str(b), "--compaction-aware", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "compaction_aware"
    assert data["compaction"]["boundary_indices"] == [3, 3]
    assert data["verdict"] == "stable_after_normalization"


def test_cli_custom_compaction_type(write_trace, capsys):
    a_steps = run_a()
    b_steps = run_b()
    for steps in (a_steps, b_steps):
        steps[3] = {"type": "squash", "summary": steps[3]["summary"]}
    a = write_trace("a.jsonl", a_steps)
    b = write_trace("b.jsonl", b_steps)
    # Unknown type: falls back to strict, and the runs diverge.
    assert main([str(a), str(b), "--compaction-aware"]) == 1
    capsys.readouterr()
    # Registered type: the boundary is honored.
    assert (
        main([str(a), str(b), "--compaction-aware", "--compaction-type", "squash"])
        == 0
    )
    assert "compaction-aware comparison:" in capsys.readouterr().out
