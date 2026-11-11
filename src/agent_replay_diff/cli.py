"""Command-line interface: argument parsing and comparison orchestration."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence

from . import __version__
from .compaction import compare_traces_compaction_aware
from .differ import compare_traces
from .loader import TraceError, load_trace
from .normalize import Normalizer
from .report import render_json, render_text

EXIT_EQUIVALENT = 0
EXIT_DIVERGENT = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-replay-diff",
        description=(
            "Compare repeated runs of the same agent task. Volatile noise "
            "(timestamps, UUIDs, request IDs, latencies, scalar-list order) "
            "is normalized away, then the runs are diffed step by step to "
            "find the first real divergence."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"agent-replay-diff {__version__}"
    )
    parser.add_argument(
        "traces",
        nargs="+",
        metavar="TRACE",
        help="two or more trace files (JSONL, or a JSON array of steps)",
    )
    parser.add_argument(
        "--ignore-key",
        action="append",
        default=[],
        metavar="KEY",
        help="extra field name whose value is volatile (repeatable, "
        "case-insensitive)",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help="extra regex for volatile content in string values (repeatable)",
    )
    parser.add_argument(
        "--no-sort-lists",
        action="store_true",
        help="do not sort scalar lists during normalization "
        "(treat list order as meaningful)",
    )
    parser.add_argument(
        "--exclude-harness-terminated",
        action="store_true",
        help="exclude runs whose final step is a harness failure "
        "(content-filter crash, timeout, provider error) from the "
        "comparison; such runs are always listed in the report, and "
        "exclusion applies only when at least two other runs remain "
        "(strict mode only)",
    )
    parser.add_argument(
        "--compaction-aware",
        action="store_true",
        help="treat each run's first compaction step (type \"compaction\" "
        "or \"context_compaction\") as an expected divergence boundary: "
        "strict comparison before it, summary spot-check at it, and "
        "tool-call-sequence + final-step comparison after it",
    )
    parser.add_argument(
        "--compaction-type",
        action="append",
        default=[],
        metavar="TYPE",
        help="additional step type that marks a compaction step "
        "(repeatable, case-insensitive)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as machine-readable JSON",
    )
    parser.add_argument(
        "--max-diff-lines",
        type=int,
        default=80,
        metavar="N",
        help="cap the first-divergence diff at N lines (default: 80)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.traces) < 2:
        parser.error("at least two trace files are required")
    if args.max_diff_lines <= 0:
        parser.error("--max-diff-lines must be a positive integer")

    patterns: list[re.Pattern[str]] = []
    for raw in args.pattern:
        try:
            patterns.append(re.compile(raw))
        except re.error as exc:
            print(f"error: invalid --pattern regex {raw!r}: {exc}", file=sys.stderr)
            return EXIT_ERROR

    try:
        traces = [load_trace(path) for path in args.traces]
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    normalizer = Normalizer(
        ignore_keys=args.ignore_key,
        patterns=patterns,
        sort_lists=not args.no_sort_lists,
    )
    if args.compaction_aware:
        report = compare_traces_compaction_aware(
            traces,
            labels=[str(path) for path in args.traces],
            normalizer=normalizer,
            max_diff_lines=args.max_diff_lines,
            extra_types=tuple(args.compaction_type),
        )
    else:
        report = compare_traces(
            traces,
            labels=[str(path) for path in args.traces],
            normalizer=normalizer,
            max_diff_lines=args.max_diff_lines,
            exclude_harness_terminated=args.exclude_harness_terminated,
        )
    print(render_json(report) if args.json else render_text(report))
    return report.exit_code
