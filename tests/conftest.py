"""Shared helpers for writing trace files in tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def write_trace(tmp_path: Path):
    """Return a helper that writes steps to a trace file and returns its path."""

    def _write(name: str, steps: list, fmt: str = "jsonl") -> Path:
        path = tmp_path / name
        if fmt == "jsonl":
            body = "\n".join(json.dumps(step) for step in steps)
            path.write_text(body + "\n" if body else "", encoding="utf-8")
        elif fmt == "json":
            path.write_text(json.dumps(steps, indent=2), encoding="utf-8")
        else:  # raw text, for malformed-input tests
            path.write_text(steps, encoding="utf-8")  # type: ignore[arg-type]
        return path

    return _write


def step(**fields) -> dict:
    """Build a minimal trace step dict from keyword fields."""
    return dict(fields)
