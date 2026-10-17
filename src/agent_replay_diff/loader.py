"""Loading recorded agent traces from disk.

A trace is the recording of one run of an agent task: a sequence of steps.
Two on-disk shapes are supported:

- **JSONL** — one JSON object per line, one step per line.
- **JSON array** — a single JSON document containing an array of step
  objects.

A step is a JSON object. By convention it carries fields such as::

    {"step": 0, "type": "tool_call", "tool": "search",
     "input": {"query": "..."}, "output": {...}}

but the loader is deliberately lenient: any subset of fields is accepted and
unknown fields are preserved untouched — the differ compares whatever is
there. What is *not* accepted: a step that is not a JSON object, malformed
JSON, or an empty trace. Those raise :class:`TraceError`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TraceError(ValueError):
    """A trace file could not be read or parsed."""


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load one trace file, returning its steps as a list of dicts."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TraceError(f"cannot read trace file {file_path}: {exc}") from exc

    stripped = text.strip()
    if not stripped:
        raise TraceError(f"trace file {file_path} is empty (no steps)")

    if stripped.startswith("["):
        steps = _parse_json_array(stripped, file_path)
    else:
        steps = _parse_jsonl(text, file_path)

    if not steps:
        raise TraceError(f"trace file {file_path} contains no steps")
    return steps


def _check_step(value: Any, where: str, file_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceError(
            f"{file_path}: {where} is not a JSON object "
            f"(got {type(value).__name__}); every step must be an object"
        )
    return value


def _parse_json_array(text: str, file_path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TraceError(f"{file_path}: invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise TraceError(f"{file_path}: top-level JSON value is not an array")
    return [
        _check_step(item, f"element {i}", file_path) for i, item in enumerate(data)
    ]


def _parse_jsonl(text: str, file_path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue  # blank lines are ignored
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceError(
                f"{file_path}: invalid JSON on line {lineno}: {exc}"
            ) from exc
        steps.append(_check_step(value, f"line {lineno}", file_path))
    return steps
