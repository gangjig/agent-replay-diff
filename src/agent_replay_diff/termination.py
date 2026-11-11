"""Detecting runs that were ended by the harness, not by the agent.

Some runs never get to finish: the provider's content filter crashes the
call, the request times out, the provider errors out — and the harness
then writes *something* as the final answer (or nothing at all). That
final answer was not produced by the agent, so counting the run as an
agent-behavior divergence misreads a harness failure as agent
instability. (In the failure analysis of *Dependable Long-Horizon LLM
Agents*, arXiv:2611.13254, 45% of human-annotated failures on one
benchmark ended exactly this way.)

Detection is deliberately conservative and looks at the **final step
only** — an error mid-trace is normal agent life (a tool call that
failed and was retried); only a run whose last recorded step is a
harness failure signal counts. Signals, in order:

- an explicit ``harness_terminated: true`` field on the final step;
- ``terminated_by: "harness"`` on the final step;
- a final-step ``type`` naming a harness failure (``harness_error``,
  ``content_filter``, ``provider_error``, ``timeout``, ...);
- a final-step ``error`` whose text names a harness cause (content
  filter, timeout, rate limit, provider/connection failure).

A bare ``type: "error"`` final step whose message names no harness
cause is *not* flagged: it may well be the agent's own failure, and
mislabeling it would hide a real divergence.
"""

from __future__ import annotations

import re
from typing import Any

#: Final-step ``type`` values that name a harness-side failure.
HARNESS_STEP_TYPES = frozenset(
    {
        "harness_error",
        "content_filter",
        "content_filter_error",
        "provider_error",
        "api_error",
        "timeout",
        "timeout_error",
        "harness_timeout",
        "rate_limit",
        "rate_limit_error",
    }
)

#: (name, pattern) pairs matched against the final step's error text.
ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("content filter", re.compile(r"content[ _-]?filter|content policy", re.I)),
    ("timeout", re.compile(r"timed? ?out|timeout", re.I)),
    ("rate limit", re.compile(r"rate[ _-]?limit|too many requests", re.I)),
    (
        "provider error",
        re.compile(
            r"provider error|overloaded|connection error|service unavailable",
            re.I,
        ),
    ),
)


def _error_text(step: dict[str, Any]) -> str:
    """Best-effort text of a step's error: string, or dict message/type/code."""
    error = step.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        parts = [
            str(error[key])
            for key in ("message", "type", "code")
            if error.get(key) is not None
        ]
        return " ".join(parts)
    error_type = step.get("error_type")
    return str(error_type) if error_type is not None else ""


def detect_harness_termination(
    trace: list[dict[str, Any]],
) -> tuple[int, str] | None:
    """Return ``(step_index, reason)`` if the run ended by harness failure.

    Only the final step is examined; returns ``None`` when the run's
    ending carries no harness-failure signal.
    """
    if not trace:
        return None
    index = len(trace) - 1
    step = trace[index]

    if step.get("harness_terminated") is True:
        return index, "final step is marked harness_terminated"

    terminated_by = step.get("terminated_by")
    if isinstance(terminated_by, str) and terminated_by.lower() == "harness":
        return index, "final step says terminated_by=harness"

    step_type = step.get("type")
    if isinstance(step_type, str) and step_type.lower() in HARNESS_STEP_TYPES:
        return index, f"final step type is {step_type!r}"

    text = _error_text(step)
    if text:
        for name, pattern in ERROR_PATTERNS:
            if pattern.search(text):
                return index, f"final step error names a {name}"
    return None
