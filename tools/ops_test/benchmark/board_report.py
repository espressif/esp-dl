#!/usr/bin/env python3
"""Quantify how much the CI boards of one chip differ from each other.

The target-test runners drive several boards of the same chip from a single
host and hand a free USB slot to each job, so an operator test shard is
measured on an arbitrary board every pipeline. Since schema v3 every result
carries the DUT's factory eFuse MAC, which makes that spread measurable:
this script reads one or more perf-results/baseline JSONs and reports, per
chip, how each board compares to the per-case median across all boards.

Only cases that were measured on at least two boards can be compared, so run
this against an accumulated baseline (the artifact of
publish_espdl_ops_perf_baseline) rather than a single pipeline's results.

Example:
    python3 board_report.py esp32p4_perf_baseline.json
    python3 board_report.py ops_perf/esp32s3/**/perf_results.json --min-us 50
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Statuses whose timings must not be used.
BAD_STATUSES = {"error", "failed", "skip", "skipped", "timeout"}


def compared_us(result):
    """The timing the perf gate would compare, or None when unusable."""
    for field in ("min_us", "median_us", "mean_us"):
        value = result.get(field)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def load(paths):
    """Return {target: {case_key: {board: us}}} from the given JSON files."""
    samples = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        for result in document.get("results", []):
            if str(result.get("status", "")).lower() in BAD_STATUSES:
                continue
            board = result.get("board")
            if not board or board == "unknown":
                continue
            value = compared_us(result)
            if value is None:
                continue
            case = (
                str(result.get("idf_version")),
                str(result.get("config")),
                str(result.get("name")),
            )
            # A board may appear more than once for a case across pipelines;
            # keep its fastest sample, which is the noise-free estimate.
            previous = samples[str(result.get("target"))][case].get(board)
            if previous is None or value < previous:
                samples[str(result.get("target"))][case][board] = value
    return samples


def analyse(cases, min_us):
    """Per-board deviations plus the per-case spread across boards."""
    deviations = defaultdict(list)
    spreads = []
    compared = 0
    for case, boards in sorted(cases.items()):
        if len(boards) < 2:
            continue
        if min_us and min(boards.values()) < min_us:
            continue
        compared += 1
        median = statistics.median(boards.values())
        for board, value in boards.items():
            deviations[board].append((value - median) / median * 100.0)
        low = min(boards.values())
        high = max(boards.values())
        spreads.append(((high - low) / low * 100.0, case, low, high))
    spreads.sort(reverse=True)
    return deviations, spreads, compared


def percentile(values, fraction):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def report(target, cases, min_us, worst):
    deviations, spreads, compared = analyse(cases, min_us)
    print("=" * 72)
    print(
        "target {}: {} case(s) total, {} comparable across boards".format(
            target, len(cases), compared
        )
    )
    if not compared:
        print(
            "  Not enough data yet: a case must be measured on at least two "
            "boards. Accumulate more pipelines and retry."
        )
        return
    print("  boards: {}".format(", ".join(sorted(deviations))))
    print()
    print("  Per-board deviation from the per-case median across boards:")
    print(
        "  {:<14} {:>6} {:>9} {:>9} {:>9}".format(
            "board", "cases", "median", "p90|abs|", "max|abs|"
        )
    )
    for board in sorted(deviations):
        values = deviations[board]
        magnitudes = [abs(value) for value in values]
        print(
            "  {:<14} {:>6} {:>8.2f}% {:>8.2f}% {:>8.2f}%".format(
                board,
                len(values),
                statistics.median(values),
                percentile(magnitudes, 0.9),
                max(magnitudes),
            )
        )
    print()
    spread_values = [spread for spread, _, _, _ in spreads]
    print("  Per-case spread (slowest vs fastest board):")
    print(
        "    median {:.2f}%, p90 {:.2f}%, p99 {:.2f}%, max {:.2f}%".format(
            statistics.median(spread_values),
            percentile(spread_values, 0.9),
            percentile(spread_values, 0.99),
            max(spread_values),
        )
    )
    print()
    print("  Worst {} case(s):".format(min(worst, len(spreads))))
    for spread, case, low, high in spreads[:worst]:
        idf_version, config, name = case
        print(
            "    {:>7.2f}%  {:<10} {:<14} {} ({:.1f} -> {:.1f} us)".format(
                spread, config, "idf " + idf_version, name, low, high
            )
        )
    print()
    print(
        "  A perf gate threshold below the p90 spread will fail on board "
        "changes alone whenever a case moves to another board."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "jsons",
        nargs="+",
        metavar="RESULTS_JSON",
        help="perf results or baseline JSON files (schema v3 or newer)",
    )
    parser.add_argument(
        "--min-us",
        type=float,
        default=0.0,
        help="ignore cases faster than this, where timer quantization "
        "dominates the comparison",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=15,
        help="how many of the worst cases to list per target",
    )
    args = parser.parse_args(argv)

    missing = [path for path in args.jsons if not Path(path).is_file()]
    if missing:
        raise SystemExit("error: file(s) not found: {}".format(", ".join(missing)))

    samples = load(args.jsons)
    if not samples:
        raise SystemExit(
            "error: no usable results with a board field found. The inputs are "
            "probably older than schema v3."
        )
    for target in sorted(samples):
        report(target, samples[target], args.min_us, args.worst)


if __name__ == "__main__":
    main()
