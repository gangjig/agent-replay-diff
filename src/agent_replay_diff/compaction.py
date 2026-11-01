"""Compaction-aware comparison: stop flagging the summarizer as a fork.

Modern harnesses compact their context mid-run: when the conversation
gets long, the earlier steps are replaced by a generated summary and
the agent continues from the summary. OpenAI's Agents SDK does this
in the background, on by default; other harnesses do the same. For a
step-by-step differ this is poison: the two runs' summaries are
*never* worded alike, and every message after the compaction quotes
the summary's wording, so a strict comparison reports a "divergence"
at the compaction step of two runs that behaved identically. The
most common fork becomes a fake one.

Compaction-aware mode treats the first compaction step of each run
as an **expected divergence boundary** and compares in three phases:

1. **Prefix (strict).** Steps before the earliest boundary are
   compared exactly as in strict mode — identical / noise-only /
   divergent, first divergence reported as usual. Before the summary
   exists, there is no excuse for drift.
2. **Boundary (spot-check).** Each run's compaction summary is
   checked, not diffed: the summaries' lengths must be within
   ``MAX_SUMMARY_LENGTH_RATIO`` of each other, and the *key entities*
   they mention (dotted identifiers like ``auth.py``, ``snake_case``
   names, numbers, quoted spans) must overlap with Jaccard similarity
   at least ``MIN_ENTITY_OVERLAP``. A summary that dropped the file
   under repair, or ballooned to ten times the length, is a real
   divergence and is reported at the boundary.
3. **Tail (behavior only).** After its own boundary, a run's message
   wording is allowed to drift — it is quoting a differently-worded
   summary. What is still compared: the **tool-call sequence**
   (tool names with normalized inputs, in order) and the **final
   step** (after standard normalization). Everything else in the
   tail is counted as ignored, and the count is reported.

Semantics worth knowing:

- The mode applies only when **every** run contains a compaction
  step. If some run never compacted, the runs genuinely took
  different paths, and the comparison falls back to strict mode
  (the report carries a note saying so).
- Runs may compact at different steps. Steps between the earliest
  boundary and a run's own boundary are covered only by that run's
  summary spot-check — the summary is supposed to be their record.
- Only the **first** compaction per run is a boundary. Later
  compaction steps sit in the tail and are ignored like other
  non-tool steps; checking every re-compaction is future work.

A step counts as a compaction step when its ``type`` is
``"compaction"`` or ``"context_compaction"`` (case-insensitive);
harnesses that log it under another name can add types via
``extra_types`` / the CLI's ``--compaction-type``.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from .differ import (
    DIVERGENT,
    NOISE_ONLY,
    VERDICT_DIVERGENT,
    VERDICT_EQUIVALENT,
    VERDICT_STABLE,
    CompactionReport,
    DiffReport,
    PairDiff,
    _pretty,
    _truncate,
    compare_traces,
)
from .normalize import Normalizer

DEFAULT_COMPACTION_TYPES = ("compaction", "context_compaction")
MAX_SUMMARY_LENGTH_RATIO = 2.0
MIN_ENTITY_OVERLAP = 0.5

_QUOTED_RE = re.compile(r"'([^'\n]{2,})'|\"([^\"\n]{2,})\"")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def is_compaction_step(
    step: dict[str, Any], types: tuple[str, ...] = DEFAULT_COMPACTION_TYPES
) -> bool:
    step_type = step.get("type")
    return isinstance(step_type, str) and step_type.lower() in types


def first_compaction_index(
    trace: list[dict[str, Any]],
    types: tuple[str, ...] = DEFAULT_COMPACTION_TYPES,
) -> int | None:
    for index, step in enumerate(trace):
        if is_compaction_step(step, types):
            return index
    return None


def summary_text(step: dict[str, Any]) -> str:
    """The summary carried by a compaction step.

    Looks for a string under ``summary``, then ``text``, then
    ``content`` — the field names harnesses actually use. Returns ""
    when the step carries none of them.
    """
    for key in ("summary", "text", "content"):
        value = step.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_entities(text: str) -> set[str]:
    """Key entities a summary mentions: dotted identifiers (file and
    module paths), snake_case names, numbers, and quoted spans."""
    entities: set[str] = set()
    for match in _QUOTED_RE.finditer(text):
        entities.add((match.group(1) or match.group(2)).strip().lower())
    for regex in (_DOTTED_RE, _SNAKE_RE, _NUMBER_RE):
        for match in regex.finditer(text):
            entities.add(match.group(0).lower())
    return entities


def _entity_overlap(summaries: list[str]) -> float:
    sets = [extract_entities(summary) for summary in summaries]
    union = set().union(*sets) if sets else set()
    if not union:
        return 1.0  # no entities anywhere: nothing to disagree about
    common = set.intersection(*sets) if sets else set()
    return len(common) / len(union)


def _length_ratio(lengths: list[int]) -> float:
    smallest, largest = min(lengths), max(lengths)
    if smallest == 0:
        return 1.0 if largest == 0 else float("inf")
    return largest / smallest


def _tool_call_sequence(
    tail: list[dict[str, Any]], normalizer: Normalizer
) -> list[tuple[str, str]]:
    sequence: list[tuple[str, str]] = []
    for step in tail:
        if step.get("type") == "tool_call":
            normalized_input = normalizer.normalize(step.get("input"))
            sequence.append(
                (
                    str(step.get("tool")),
                    json.dumps(
                        normalized_input, sort_keys=True, ensure_ascii=False,
                        default=str,
                    ),
                )
            )
    return sequence


def _first_sequence_mismatch(
    sequences: list[list[tuple[str, str]]], labels: list[str]
) -> str | None:
    baseline = sequences[0]
    for run_no in range(1, len(sequences)):
        other = sequences[run_no]
        for position in range(max(len(baseline), len(other))):
            base_entry = baseline[position] if position < len(baseline) else None
            other_entry = other[position] if position < len(other) else None
            if base_entry != other_entry:
                base_desc = (
                    f"{base_entry[0]} {base_entry[1]}"
                    if base_entry
                    else "(sequence ended)"
                )
                other_desc = (
                    f"{other_entry[0]} {other_entry[1]}"
                    if other_entry
                    else "(sequence ended)"
                )
                return (
                    f"post-compaction tool call #{position + 1}: "
                    f"{labels[0]} has {base_desc}, "
                    f"{labels[run_no]} has {other_desc}"
                )
    return None


def compare_traces_compaction_aware(
    traces: list[list[dict[str, Any]]],
    labels: list[str] | None = None,
    normalizer: Normalizer | None = None,
    max_diff_lines: int = 80,
    extra_types: tuple[str, ...] = (),
) -> DiffReport:
    """Compare traces with the first compaction step per run treated
    as an expected divergence boundary. See the module docstring."""
    if len(traces) < 2:
        raise ValueError("at least two traces are required to compare")
    if labels is None:
        labels = [f"run {i + 1}" for i in range(len(traces))]
    if len(labels) != len(traces):
        raise ValueError("labels must match the number of traces")
    normalizer = normalizer or Normalizer()
    types = DEFAULT_COMPACTION_TYPES + tuple(
        str(t).lower() for t in extra_types
    )

    boundaries = [first_compaction_index(trace, types) for trace in traces]
    if any(boundary is None for boundary in boundaries):
        missing = [
            labels[i] for i, boundary in enumerate(boundaries) if boundary is None
        ]
        report = compare_traces(
            traces, labels=labels, normalizer=normalizer,
            max_diff_lines=max_diff_lines,
        )
        report.compaction_note = (
            "compaction-aware mode requested but not applied: "
            + ", ".join(missing)
            + " contain(s) no compaction step; all steps compared strictly"
        )
        return report

    boundary_indices = [int(b) for b in boundaries if b is not None]
    boundary = min(boundary_indices)

    # Phase 1: strict comparison of the prefixes.
    prefix_report = compare_traces(
        [trace[:boundary] for trace in traces],
        labels=labels,
        normalizer=normalizer,
        max_diff_lines=max_diff_lines,
    )

    # Phase 2: spot-check the summaries.
    summaries = [
        summary_text(trace[b]) for trace, b in zip(traces, boundary_indices)
    ]
    summary_lengths = [len(summary) for summary in summaries]
    ratio = _length_ratio(summary_lengths)
    overlap = _entity_overlap(summaries)
    summary_ok = ratio <= MAX_SUMMARY_LENGTH_RATIO and overlap >= MIN_ENTITY_OVERLAP

    # Phase 3: behavior-only comparison of the tails.
    tails = [
        trace[b + 1:] for trace, b in zip(traces, boundary_indices)
    ]
    sequences = [_tool_call_sequence(tail, normalizer) for tail in tails]
    tail_mismatch = _first_sequence_mismatch(sequences, labels)

    final_compared = False
    final_equal: bool | None = None
    ignored_steps = 0
    for tail in tails:
        tool_calls = sum(1 for step in tail if step.get("type") == "tool_call")
        ignored_steps += len(tail) - tool_calls
    finals = []
    for tail in tails:
        if tail:
            finals.append(normalizer.normalize(tail[-1]))
    if finals and len(finals) == len(tails):
        final_compared = True
        final_equal = all(final == finals[0] for final in finals[1:])
        # The final step was compared, not ignored — unless it is a
        # tool call, in which case it was already counted as one.
        for tail in tails:
            if tail and tail[-1].get("type") != "tool_call":
                ignored_steps -= 1

    compaction = CompactionReport(
        boundary_indices=boundary_indices,
        summary_lengths=summary_lengths,
        summary_length_ratio=ratio,
        summary_entity_overlap=overlap,
        summary_ok=summary_ok,
        tail_tool_calls=[len(sequence) for sequence in sequences],
        tail_tool_mismatch=tail_mismatch,
        final_compared=final_compared,
        final_equal=final_equal,
        ignored_steps=ignored_steps,
    )

    # Verdict assembly: an earlier strict divergence wins; then the
    # boundary; then the tail.
    prefix_diverged = prefix_report.counts[DIVERGENT] > 0
    tail_diverged = tail_mismatch is not None or final_equal is False
    divergence_diffs = list(prefix_report.divergence_diffs)
    if prefix_diverged:
        verdict = VERDICT_DIVERGENT
        first_divergence = prefix_report.first_divergence
    elif not summary_ok or tail_diverged:
        verdict = VERDICT_DIVERGENT
        first_divergence = boundary
        if not summary_ok:
            # Show the summaries that failed the spot-check.
            base_step = traces[0][boundary_indices[0]]
            for run_no in range(1, len(traces)):
                other_step = traces[run_no][boundary_indices[run_no]]
                diff_lines = list(
                    difflib.unified_diff(
                        _pretty(base_step),
                        _pretty(other_step),
                        fromfile=(
                            f"{labels[0]} step {boundary_indices[0]} "
                            f"(compaction, raw)"
                        ),
                        tofile=(
                            f"{labels[run_no]} step {boundary_indices[run_no]} "
                            f"(compaction, raw)"
                        ),
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
    else:
        first_divergence = None
        summaries_differ = len(set(summaries)) > 1
        if (
            prefix_report.counts[NOISE_ONLY] > 0
            or ignored_steps > 0
            or summaries_differ
        ):
            verdict = VERDICT_EQUIVALENT
        else:
            verdict = VERDICT_STABLE

    return DiffReport(
        labels=list(labels),
        step_counts=[len(trace) for trace in traces],
        steps=prefix_report.steps,
        first_divergence=first_divergence,
        divergence_diffs=divergence_diffs,
        mode="compaction_aware",
        compaction=compaction,
        verdict_override=verdict,
    )
