"""Step-by-step comparison of two or more recorded runs of one agent task.

Runs are aligned **by step index** (step *i* of run A is compared with step
*i* of run B). Each aligned step is classified:

- ``identical``  — the raw steps are structurally equal in every run.
- ``noise_only`` — the raw steps differ, but the differences vanish after
  normalization (timestamps, IDs, latencies, scalar-list reordering, ...).
- ``divergent``  — the normalized steps still differ, or a step is missing
  from some run: the agent actually did something different here.

The report names the *first* divergent step and carries a readable
unified-style diff of the raw step content there, which is the place a human
usually wants to start debugging flaky harness behavior.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any

from .normalize import Normalizer

IDENTICAL = "identical"
NOISE_ONLY = "noise_only"
DIVERGENT = "divergent"

VERDICT_STABLE = "stable"
VERDICT_EQUIVALENT = "stable_after_normalization"
VERDICT_DIVERGENT = "divergent"

_MISSING: Any = object()


@dataclass(frozen=True)
class StepComparison:
    index: int
    classification: str  # IDENTICAL | NOISE_ONLY | DIVERGENT
    present_in_all: bool


@dataclass(frozen=True)
class PairDiff:
    """A raw-content diff of the first divergent step, baseline vs one run."""

    baseline_label: str
    other_label: str
    diff: str  # unified diff text


@dataclass
class DiffReport:
    labels: list[str]
    step_counts: list[int]
    steps: list[StepComparison]
    first_divergence: int | None
    divergence_diffs: list[PairDiff] = field(default_factory=list)

    @property
    def compared_steps(self) -> int:
        return len(self.steps)

    @property
    def counts(self) -> dict[str, int]:
        result = {IDENTICAL: 0, NOISE_ONLY: 0, DIVERGENT: 0}
        for step in self.steps:
            result[step.classification] += 1
        return result

    @property
    def verdict(self) -> str:
        counts = self.counts
        if counts[DIVERGENT]:
            return VERDICT_DIVERGENT
        if counts[NOISE_ONLY]:
            return VERDICT_EQUIVALENT
        return VERDICT_STABLE

    @property
    def exit_code(self) -> int:
        """0 when the runs are equivalent (noise-only differences are OK)."""
        return 1 if self.verdict == VERDICT_DIVERGENT else 0

    def to_dict(self) -> dict[str, Any]:
        counts = self.counts
        return {
            "runs": [
                {"label": label, "steps": count}
                for label, count in zip(self.labels, self.step_counts)
            ],
            "compared_steps": self.compared_steps,
            "counts": counts,
            "verdict": self.verdict,
            "first_divergence": self.first_divergence,
            "steps": [
                {
                    "index": step.index,
                    "classification": step.classification,
                    "present_in_all": step.present_in_all,
                }
                for step in self.steps
            ],
            "divergence_diffs": [
                {
                    "baseline": pair.baseline_label,
                    "other": pair.other_label,
                    "diff": pair.diff,
                }
                for pair in self.divergence_diffs
            ],
        }


def _pretty(step: Any) -> list[str]:
    """Canonical pretty-printed lines of a raw step, for diffing."""
    if step is _MISSING:
        return ["<missing step>"]
    text = json.dumps(step, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    return text.splitlines()


def _truncate(lines: list[str], max_lines: int) -> list[str]:
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    kept.append(f"... (diff truncated: {len(lines) - max_lines} more lines)")
    return kept


def compare_traces(
    traces: list[list[dict[str, Any]]],
    labels: list[str] | None = None,
    normalizer: Normalizer | None = None,
    max_diff_lines: int = 80,
) -> DiffReport:
    """Compare 2+ traces of the same task and build a :class:`DiffReport`."""
    if len(traces) < 2:
        raise ValueError("at least two traces are required to compare")
    if labels is None:
        labels = [f"run {i + 1}" for i in range(len(traces))]
    if len(labels) != len(traces):
        raise ValueError("labels must match the number of traces")
    normalizer = normalizer or Normalizer()

    normalized = [
        [normalizer.normalize(step) for step in trace] for trace in traces
    ]
    total_steps = max(len(trace) for trace in traces)

    steps: list[StepComparison] = []
    first_divergence: int | None = None
    for index in range(total_steps):
        raws = [
            trace[index] if index < len(trace) else _MISSING for trace in traces
        ]
        present_in_all = all(raw is not _MISSING for raw in raws)
        if present_in_all and all(raw == raws[0] for raw in raws[1:]):
            classification = IDENTICAL
        elif present_in_all and all(
            norm[index] == normalized[0][index] for norm in normalized[1:]
        ):
            classification = NOISE_ONLY
        else:
            classification = DIVERGENT
            if first_divergence is None:
                first_divergence = index
        steps.append(
            StepComparison(
                index=index,
                classification=classification,
                present_in_all=present_in_all,
            )
        )

    divergence_diffs: list[PairDiff] = []
    if first_divergence is not None:
        index = first_divergence
        base_raw = traces[0][index] if index < len(traces[0]) else _MISSING
        base_norm = (
            normalized[0][index] if index < len(normalized[0]) else _MISSING
        )
        for run_no in range(1, len(traces)):
            other_raw = (
                traces[run_no][index] if index < len(traces[run_no]) else _MISSING
            )
            other_norm = (
                normalized[run_no][index]
                if index < len(normalized[run_no])
                else _MISSING
            )
            if other_norm == base_norm:
                # Agrees with the baseline after normalization — either
                # identical or noise-only; the split at this step is
                # between the baseline and the runs that got a diff.
                continue
            diff_lines = list(
                difflib.unified_diff(
                    _pretty(base_raw),
                    _pretty(other_raw),
                    fromfile=f"{labels[0]} step {index} (raw)",
                    tofile=f"{labels[run_no]} step {index} (raw)",
                    lineterm="",
                )
            )
            divergence_diffs.append(
                PairDiff(
                    baseline_label=labels[0],
                    other_label=labels[run_no],
                    diff="\n".join(_truncate(diff_lines, max_diff_lines)),
                )
            )

    return DiffReport(
        labels=list(labels),
        step_counts=[len(trace) for trace in traces],
        steps=steps,
        first_divergence=first_divergence,
        divergence_diffs=divergence_diffs,
    )
