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

    if report.harness_terminations:
        lines.append("harness-terminated runs (ending was a harness failure):")
        for term in report.harness_terminations:
            scope = (
                "excluded from comparison"
                if term.excluded
                else "included in comparison"
            )
            lines.append(
                f"  {term.label}: step {term.step_index} — {term.reason} ({scope})"
            )
        lines.append("")

    if report.termination_note:
        lines.append(f"note: {report.termination_note}")
        lines.append("")

    if report.compaction_note:
        lines.append(f"note: {report.compaction_note}")
        lines.append("")

    if report.compaction is not None:
        c = report.compaction
        boundaries = ", ".join(
            f"{label} step {index}"
            for label, index in zip(report.labels, c.boundary_indices)
        )
        ratio = (
            "inf"
            if c.summary_length_ratio == float("inf")
            else f"{c.summary_length_ratio:.2f}"
        )
        lines.append("compaction-aware comparison:")
        lines.append(f"  boundary (first compaction): {boundaries}")
        lines.append(
            "  summaries: lengths "
            + "/".join(str(n) for n in c.summary_lengths)
            + f" chars (ratio {ratio}), entity overlap "
            f"{c.summary_entity_overlap:.2f} — "
            + ("OK" if c.summary_ok else "MISMATCH")
        )
        if c.final_compared:
            final_text = "equal" if c.final_equal else "DIFFERS"
        else:
            final_text = "not compared (a run ends at its boundary)"
        lines.append(
            f"  tail: tool calls per run {c.tail_tool_calls}, "
            f"final step {final_text}, "
            f"{c.ignored_steps} post-compaction step(s) ignored"
        )
        if c.tail_tool_mismatch:
            lines.append(f"  tail mismatch: {c.tail_tool_mismatch}")
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
