"""Normalization of volatile content in recorded agent traces.

Two runs of the same agent task almost never produce byte-identical traces:
every tool call carries fresh timestamps, request IDs, UUIDs and latency
measurements. Those differences are *noise* — they say nothing about whether
the agent behaved the same way. This module rewrites a trace step into a
canonical form where volatile content is replaced by placeholders, so two
steps can be compared for *meaningful* equality.

What is normalized
------------------
- Values under volatile keys (see ``DEFAULT_VOLATILE_KEYS``, plus any key
  ending in ``_at`` or ``_ms``, plus user-supplied ``ignore_keys``) are
  replaced wholesale with ``"<volatile>"``. The key itself is kept, so a step
  that *gains or loses* a volatile field still differs structurally.
- Inside strings: ISO-8601 datetimes, dates, times, UUIDs, long hex strings
  (32+ chars, e.g. hashes), and epoch timestamps (10-digit seconds /
  13-digit milliseconds, as text or as numbers) are replaced with
  placeholders — including when embedded in longer text.
- User-supplied regex patterns are applied (in order, before the built-ins)
  and their matches replaced with ``"<custom>"``.
- Dict keys are emitted in sorted order (canonical form), and lists whose
  elements are all scalars are sorted, since tool results frequently return
  the same set in arbitrary order. ``sort_lists=False`` disables the sorting.
  Lists containing objects are never reordered: their order may be semantic.

Normalization never mutates its input; it returns a new structure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

TIMESTAMP_PLACEHOLDER = "<timestamp>"
DATE_PLACEHOLDER = "<date>"
TIME_PLACEHOLDER = "<time>"
UUID_PLACEHOLDER = "<uuid>"
HEX_PLACEHOLDER = "<hex>"
EPOCH_PLACEHOLDER = "<epoch>"
CUSTOM_PLACEHOLDER = "<custom>"
VOLATILE_PLACEHOLDER = "<volatile>"

#: Key names (compared case-insensitively) whose values are always volatile.
DEFAULT_VOLATILE_KEYS = frozenset(
    {
        "timestamp",
        "ts",
        "time",
        "date",
        "created_at",
        "updated_at",
        "started_at",
        "ended_at",
        "start_time",
        "end_time",
        "duration",
        "duration_ms",
        "duration_s",
        "latency",
        "latency_ms",
        "elapsed",
        "elapsed_ms",
        "elapsed_s",
        "elapsed_time",
        "response_time",
        "response_time_ms",
        "took_ms",
        "request_id",
        "trace_id",
        "session_id",
        "run_id",
        "uuid",
    }
)

_KEY_SUFFIXES = ("_at", "_ms")

_ISO_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_EPOCH_MS_RE = re.compile(r"\b1\d{12}\b")
_EPOCH_S_RE = re.compile(r"\b1\d{9}\b")

# Plausible epoch ranges (roughly 2001-2033 in seconds / milliseconds).
_EPOCH_S_RANGE = (1_000_000_000, 2_000_000_000)
_EPOCH_MS_RANGE = (1_000_000_000_000, 2_000_000_000_000)

_SCALAR_TYPES = (str, int, float, bool, type(None))

# Applied in order; earlier rules win where patterns overlap.
_BUILTIN_STRING_RULES = (
    (_ISO_DATETIME_RE, TIMESTAMP_PLACEHOLDER),
    (_DATE_RE, DATE_PLACEHOLDER),
    (_TIME_RE, TIME_PLACEHOLDER),
    (_UUID_RE, UUID_PLACEHOLDER),
    (_HEX_RE, HEX_PLACEHOLDER),
    (_EPOCH_MS_RE, EPOCH_PLACEHOLDER),
    (_EPOCH_S_RE, EPOCH_PLACEHOLDER),
)


def _canonical(value: Any) -> str:
    """A deterministic string form of a JSON value, for sort keys."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class Normalizer:
    """Rewrite trace steps into canonical, noise-free form.

    Parameters
    ----------
    ignore_keys:
        Extra key names (case-insensitive) whose values are volatile.
    patterns:
        Extra regexes (strings or compiled patterns). Matches are replaced
        with ``"<custom>"`` before the built-in rules run.
    sort_lists:
        When true (default), lists of scalars are sorted after
        normalization.
    """

    def __init__(
        self,
        ignore_keys: Iterable[str] = (),
        patterns: Iterable[str | re.Pattern[str]] = (),
        sort_lists: bool = True,
    ) -> None:
        self.volatile_keys = DEFAULT_VOLATILE_KEYS | {
            str(k).lower() for k in ignore_keys
        }
        self.patterns = [
            re.compile(p) if isinstance(p, str) else p for p in patterns
        ]
        self.sort_lists = sort_lists

    def _is_volatile_key(self, key: str) -> bool:
        lowered = str(key).lower()
        return lowered in self.volatile_keys or lowered.endswith(_KEY_SUFFIXES)

    def normalize(self, value: Any) -> Any:
        """Return the normalized form of a JSON value (never mutates input)."""
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key in sorted(value, key=str):
                if self._is_volatile_key(key):
                    out[key] = VOLATILE_PLACEHOLDER
                else:
                    out[key] = self.normalize(value[key])
            return out
        if isinstance(value, list):
            items = [self.normalize(item) for item in value]
            if (
                self.sort_lists
                and len(items) > 1
                and all(isinstance(item, _SCALAR_TYPES) for item in items)
            ):
                items = sorted(items, key=_canonical)
            return items
        if isinstance(value, str):
            return self._normalize_string(value)
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return self._normalize_number(value)
        return value

    def _normalize_string(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub(CUSTOM_PLACEHOLDER, text)
        for regex, placeholder in _BUILTIN_STRING_RULES:
            text = regex.sub(placeholder, text)
        return text

    @staticmethod
    def _normalize_number(value: int | float) -> int | float | str:
        if isinstance(value, float) and not value.is_integer():
            return value
        as_int = int(value)
        if _EPOCH_S_RANGE[0] <= as_int < _EPOCH_S_RANGE[1]:
            return EPOCH_PLACEHOLDER
        if _EPOCH_MS_RANGE[0] <= as_int < _EPOCH_MS_RANGE[1]:
            return EPOCH_PLACEHOLDER
        return value
