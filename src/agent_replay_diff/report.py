"""Render a :class:`~agent_replay_diff.differ.DiffReport` as text or JSON."""

from __future__ import annotations

import json

from .differ import (
    DIVERGENT,
    IDENTICAL,
    NOISE_ONLY,
    DiffReport,
    VERDICT_DIVERGENT,
    VERDICT_EQUIVALENT,
)

_VERDICT_LINES = {
    VERDICT_DIVERGENT: "DIVERGENT",
    VERDICT_EQUIVALENT: "STABLE AFTER NORMALIZATION (differences were volatile noise only)",
}


def render_json(report: DiffReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def render_text(report: DiffReport) -> str:
    counts = report.counts
    lines: list[str] = ["agent-replay-diff report", "=" * 24]
    for label, count in zip(report.labels, report.step_counts):
        lines.append(f"{label}: {count} steps")
    lines.append(f"compared steps: {report.compared_steps}")
    lines.append("")
    lines.append(f"identical:   {counts[IDENTICAL]}")
    lines.append(f"noise-only:  {counts[NOISE_ONLY]}")
    lines.append(f"divergent:   {counts[DIVERGENT]}")
    lines.append("")

    if report.verdict == VERDICT_DIVERGENT:
        lines.append(f"verdict: {_VERDICT_LINES[VERDICT_DIVERGENT]}")
        lines.append(f"first divergence: step {report.first_divergence}")
        for pair in report.divergence_diffs:
            lines.append("")
            lines.append(pair.diff)
    elif report.verdict == VERDICT_EQUIVALENT:
        lines.append(f"verdict: {_VERDICT_LINES[VERDICT_EQUIVALENT]}")
        lines.append("first divergence: none")
    else:
        lines.append("verdict: STABLE (runs are identical at every step)")
        lines.append("first divergence: none")
    return "\n".join(lines)
