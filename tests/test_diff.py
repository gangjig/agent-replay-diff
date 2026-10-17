"""Tests for the step-by-step differ and its report."""

from __future__ import annotations

import pytest

from agent_replay_diff.differ import (
    DIVERGENT,
    IDENTICAL,
    NOISE_ONLY,
    compare_traces,
)
from agent_replay_diff.normalize import Normalizer


def msg(content, **extra):
    return {"type": "message", "content": content, **extra}


def test_identical_runs():
    traces = [[msg("a"), msg("b")], [msg("a"), msg("b")]]
    report = compare_traces(traces)
    assert report.counts == {IDENTICAL: 2, NOISE_ONLY: 0, DIVERGENT: 0}
    assert report.verdict == "stable"
    assert report.exit_code == 0
    assert report.first_divergence is None
    assert report.divergence_diffs == []


def test_noise_only_runs():
    a = [msg("a", timestamp="2026-10-17T08:00:05Z", duration_ms=10)]
    b = [msg("a", timestamp="2026-10-17T09:31:57Z", duration_ms=812)]
    report = compare_traces([a, b])
    assert report.steps[0].classification == NOISE_ONLY
    assert report.verdict == "stable_after_normalization"
    assert report.exit_code == 0
    assert report.first_divergence is None


def test_noise_from_list_order():
    a = [{"output": {"tags": ["x", "y", "z"]}}]
    b = [{"output": {"tags": ["z", "x", "y"]}}]
    report = compare_traces([a, b])
    assert report.steps[0].classification == NOISE_ONLY


def test_divergence_at_step_k():
    a = [msg("a"), msg("b"), msg("c"), msg("d")]
    b = [msg("a"), msg("b"), msg("DIFFERENT"), msg("d")]
    report = compare_traces([a, b])
    assert report.first_divergence == 2
    assert report.steps[2].classification == DIVERGENT
    assert report.steps[3].classification == IDENTICAL  # classified per step
    assert report.verdict == "divergent"
    assert report.exit_code == 1


def test_first_divergence_is_the_earliest():
    a = [msg("x"), msg("y")]
    b = [msg("p"), msg("q")]
    report = compare_traces([a, b])
    assert report.first_divergence == 0


def test_divergence_diff_shows_raw_values():
    a = [msg("same"), msg("alpha result")]
    b = [msg("same"), msg("beta result")]
    report = compare_traces([a, b])
    assert len(report.divergence_diffs) == 1
    diff = report.divergence_diffs[0].diff
    assert "alpha result" in diff
    assert "beta result" in diff
    assert diff.startswith("---")


def test_unequal_lengths_extra_step_is_divergent():
    a = [msg("a"), msg("b")]
    b = [msg("a"), msg("b"), msg("extra")]
    report = compare_traces([a, b])
    assert report.step_counts == [2, 3]
    assert report.compared_steps == 3
    assert report.steps[2].classification == DIVERGENT
    assert report.steps[2].present_in_all is False
    assert report.first_divergence == 2
    assert report.exit_code == 1


def test_missing_step_diff_marker():
    a = [msg("a")]
    b = [msg("a"), msg("b")]
    report = compare_traces([a, b])
    assert "<missing step>" in report.divergence_diffs[0].diff


def test_three_runs_all_identical():
    traces = [[msg("a")], [msg("a")], [msg("a")]]
    report = compare_traces(traces)
    assert report.verdict == "stable"
    assert report.step_counts == [1, 1, 1]


def test_three_runs_one_divergent():
    traces = [[msg("a"), msg("x")], [msg("a"), msg("x")], [msg("a"), msg("y")]]
    report = compare_traces(traces, labels=["r1", "r2", "r3"])
    assert report.first_divergence == 1
    # only the run that differs from the baseline gets a diff
    assert len(report.divergence_diffs) == 1
    assert report.divergence_diffs[0].other_label == "r3"


def test_three_runs_noise_only_in_one():
    traces = [
        [msg("a", duration_ms=1)],
        [msg("a", duration_ms=2)],
        [msg("a", duration_ms=3)],
    ]
    report = compare_traces(traces)
    assert report.steps[0].classification == NOISE_ONLY
    assert report.exit_code == 0


def test_baseline_run_is_the_odd_one():
    traces = [[msg("odd")], [msg("norm")], [msg("norm")]]
    report = compare_traces(traces, labels=["r1", "r2", "r3"])
    assert report.first_divergence == 0
    others = {pair.other_label for pair in report.divergence_diffs}
    assert others == {"r2", "r3"}


def test_max_diff_lines_truncates():
    big_a = {"output": {f"k{i}": i for i in range(50)}}
    big_b = {"output": {f"k{i}": i + 1000 for i in range(50)}}
    report = compare_traces([[big_a], [big_b]], max_diff_lines=10)
    diff = report.divergence_diffs[0].diff
    assert "diff truncated" in diff
    assert len(diff.splitlines()) <= 11


def test_single_step_traces():
    report = compare_traces([[msg("a")], [msg("a")]])
    assert report.compared_steps == 1


def test_custom_normalizer_respected():
    a = [msg("a", seed=1)]
    b = [msg("a", seed=2)]
    plain = compare_traces([a, b])
    assert plain.steps[0].classification == DIVERGENT
    with_seed = compare_traces(
        [a, b], normalizer=Normalizer(ignore_keys=["seed"])
    )
    assert with_seed.steps[0].classification == NOISE_ONLY


def test_requires_two_traces():
    with pytest.raises(ValueError):
        compare_traces([[msg("a")]])


def test_to_dict_structure():
    a = [msg("a"), msg("b", duration_ms=1)]
    b = [msg("a"), msg("b", duration_ms=2)]
    data = compare_traces([a, b], labels=["A", "B"]).to_dict()
    assert data["verdict"] == "stable_after_normalization"
    assert data["first_divergence"] is None
    assert data["compared_steps"] == 2
    assert sum(data["counts"].values()) == 2
    assert data["runs"] == [
        {"label": "A", "steps": 2},
        {"label": "B", "steps": 2},
    ]
    assert [s["classification"] for s in data["steps"]] == [
        IDENTICAL,
        NOISE_ONLY,
    ]
    assert data["divergence_diffs"] == []


def test_to_dict_divergent_includes_diffs():
    data = compare_traces([[msg("a")], [msg("b")]]).to_dict()
    assert data["verdict"] == "divergent"
    assert data["first_divergence"] == 0
    assert len(data["divergence_diffs"]) == 1
    assert "diff" in data["divergence_diffs"][0]
