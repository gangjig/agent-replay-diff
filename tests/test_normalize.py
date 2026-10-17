"""Unit tests for the volatile-content normalizer."""

from __future__ import annotations

import json

import pytest

from agent_replay_diff.normalize import (
    CUSTOM_PLACEHOLDER,
    EPOCH_PLACEHOLDER,
    HEX_PLACEHOLDER,
    TIMESTAMP_PLACEHOLDER,
    UUID_PLACEHOLDER,
    VOLATILE_PLACEHOLDER,
    Normalizer,
)

N = Normalizer()


# --- ISO-8601 datetimes, dates, times -------------------------------------


def test_iso_datetime_replaced():
    assert N.normalize("2026-10-17T08:00:05Z") == TIMESTAMP_PLACEHOLDER


def test_iso_datetime_with_fraction_and_offset():
    assert N.normalize("2026-10-17T08:00:05.412+05:30") == TIMESTAMP_PLACEHOLDER
    assert N.normalize("2026-10-17 08:00:05") == TIMESTAMP_PLACEHOLDER


def test_datetime_embedded_in_text():
    out = N.normalize("finished at 2026-10-17T08:00:05Z sharp")
    assert out == f"finished at {TIMESTAMP_PLACEHOLDER} sharp"


def test_date_only_replaced():
    assert N.normalize("2026-10-17") == "<date>"


def test_time_only_replaced():
    assert N.normalize("08:00:05") == "<time>"
    assert N.normalize("call at 08:00:05.5") == f"call at <time>"


def test_two_different_timestamps_normalize_equal():
    a = N.normalize({"at": "2026-10-17T08:00:05Z", "v": 1})
    # "at" is not a volatile key, so this exercises string replacement
    b = N.normalize({"at": "2027-01-01T23:59:59Z", "v": 1})
    assert a == b


# --- UUIDs and hex ----------------------------------------------------------


def test_uuid_replaced():
    assert (
        N.normalize("3f6b0f9e-4c2a-4e1f-9a2b-7c5d8e0f1a2b") == UUID_PLACEHOLDER
    )


def test_uuid_uppercase_replaced():
    assert (
        N.normalize("3F6B0F9E-4C2A-4E1F-9A2B-7C5D8E0F1A2B") == UUID_PLACEHOLDER
    )


def test_uuid_embedded_in_text():
    out = N.normalize("id=3f6b0f9e-4c2a-4e1f-9a2b-7c5d8e0f1a2b done")
    assert out == f"id={UUID_PLACEHOLDER} done"


def test_long_hex_replaced():
    sha = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    assert N.normalize(sha) == HEX_PLACEHOLDER


def test_exactly_32_hex_replaced():
    assert N.normalize("a" * 32) == HEX_PLACEHOLDER


def test_short_hex_untouched():
    assert N.normalize("deadbeef") == "deadbeef"
    assert N.normalize("color #a1b2c3") == "color #a1b2c3"


# --- Epoch timestamps -------------------------------------------------------


def test_epoch_seconds_string():
    assert N.normalize("1792231200") == EPOCH_PLACEHOLDER


def test_epoch_milliseconds_string():
    assert N.normalize("1792231200123") == EPOCH_PLACEHOLDER


def test_epoch_seconds_int():
    assert N.normalize(1792231200) == EPOCH_PLACEHOLDER


def test_epoch_milliseconds_int():
    assert N.normalize(1792231200123) == EPOCH_PLACEHOLDER


def test_epoch_integral_float():
    assert N.normalize(1792231200.0) == EPOCH_PLACEHOLDER


def test_ordinary_numbers_untouched():
    assert N.normalize(42) == 42
    assert N.normalize(0) == 0
    assert N.normalize(3.14) == 3.14
    assert N.normalize(1792231200.5) == 1792231200.5  # non-integral float


def test_large_non_epoch_int_untouched():
    assert N.normalize(5_000_000_000) == 5_000_000_000


def test_epoch_embedded_in_text():
    assert N.normalize("t=1792231200 end") == f"t={EPOCH_PLACEHOLDER} end"


# --- Volatile keys ------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "timestamp",
        "created_at",
        "updated_at",
        "duration_ms",
        "duration",
        "latency",
        "latency_ms",
        "elapsed",
        "elapsed_ms",
        "request_id",
        "trace_id",
        "session_id",
        "run_id",
    ],
)
def test_default_volatile_keys(key):
    out = N.normalize({key: "whatever", "v": 1})
    assert out[key] == VOLATILE_PLACEHOLDER
    assert out["v"] == 1


def test_volatile_key_suffix_rules():
    out = N.normalize({"finished_at": 1, "ttfb_ms": 2, "kept": 3})
    assert out["finished_at"] == VOLATILE_PLACEHOLDER
    assert out["ttfb_ms"] == VOLATILE_PLACEHOLDER
    assert out["kept"] == 3


def test_volatile_key_case_insensitive():
    out = N.normalize({"Duration_MS": 5, "TIMESTAMP": "x"})
    assert out["Duration_MS"] == VOLATILE_PLACEHOLDER
    assert out["TIMESTAMP"] == VOLATILE_PLACEHOLDER


def test_volatile_key_value_replaced_wholesale():
    out = N.normalize({"output": {"latency": {"p50": 10, "p99": 99}}})
    assert out["output"]["latency"] == VOLATILE_PLACEHOLDER


def test_custom_ignore_key():
    n = Normalizer(ignore_keys=["seed"])
    out = n.normalize({"seed": 1234, "v": 1})
    assert out["seed"] == VOLATILE_PLACEHOLDER


def test_custom_ignore_key_case_insensitive():
    n = Normalizer(ignore_keys=["Seed"])
    assert n.normalize({"SEED": 1})["SEED"] == VOLATILE_PLACEHOLDER


def test_non_volatile_key_untouched():
    out = N.normalize({"query": "arlington", "limit": 5})
    assert out == {"limit": 5, "query": "arlington"}


# --- Structure: recursion, key order, list sorting ---------------------------


def test_nested_structures_recursed():
    value = {"a": [{"b": [{"timestamp": "2026-10-17T00:00:00Z", "id": "x"}]}]}
    out = N.normalize(value)
    assert out["a"][0]["b"][0]["timestamp"] == VOLATILE_PLACEHOLDER
    assert out["a"][0]["b"][0]["id"] == "x"


def test_dict_keys_canonical_order():
    out = N.normalize({"z": 1, "a": 2, "m": 3})
    assert list(out.keys()) == ["a", "m", "z"]
    assert json.dumps(out, sort_keys=False) == '{"a": 2, "m": 3, "z": 1}'


def test_scalar_list_sorted():
    assert N.normalize({"v": [3, 1, 2]})["v"] == [1, 2, 3]
    assert N.normalize({"v": ["b", "a", "c"]})["v"] == ["a", "b", "c"]


def test_scalar_list_mixed_types_sorted_without_error():
    out = N.normalize({"v": [2, "a", 1]})["v"]
    assert sorted(map(str, out)) == ["1", "2", "a"]


def test_sort_lists_disabled():
    n = Normalizer(sort_lists=False)
    assert n.normalize({"v": [3, 1, 2]})["v"] == [3, 1, 2]


def test_list_of_dicts_not_sorted():
    value = {"v": [{"k": 2}, {"k": 1}]}
    assert N.normalize(value)["v"] == [{"k": 2}, {"k": 1}]


def test_mixed_list_not_sorted():
    value = {"v": [3, {"k": 1}, 1]}
    assert N.normalize(value)["v"] == [3, {"k": 1}, 1]


def test_inner_scalar_list_sorted():
    out = N.normalize({"v": [[3, 1], [2]]})
    assert out["v"] == [[1, 3], [2]]


def test_booleans_and_none_preserved():
    out = N.normalize({"a": True, "b": False, "c": None})
    assert out == {"a": True, "b": False, "c": None}


def test_input_not_mutated():
    value = {"timestamp": "2026-10-17T08:00:05Z", "v": [2, 1]}
    snapshot = json.loads(json.dumps(value))
    N.normalize(value)
    assert value == snapshot


def test_plain_string_unchanged():
    assert N.normalize("just some words") == "just some words"


# --- Custom patterns ----------------------------------------------------------


def test_custom_pattern_applied():
    n = Normalizer(patterns=[r"job-\d+"])
    assert n.normalize("job-12345") == CUSTOM_PLACEHOLDER


def test_custom_pattern_embedded_in_text():
    n = Normalizer(patterns=[r"job-\d+"])
    assert n.normalize("ran job-999 ok") == f"ran {CUSTOM_PLACEHOLDER} ok"


def test_custom_patterns_make_runs_equal():
    n = Normalizer(patterns=[r"job-\d+"])
    assert n.normalize({"o": "job-1"}) == n.normalize({"o": "job-2"})


def test_custom_pattern_accepts_compiled():
    import re

    n = Normalizer(patterns=[re.compile(r"tok_[a-z]+")])
    assert n.normalize("tok_abc") == CUSTOM_PLACEHOLDER
