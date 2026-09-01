#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-target operator benchmark report generator for ESP-DL.

Reads the per-target perf-results JSON files produced by the operator test
suite (tools/ops_test/benchmark/<target>_perf_results.json) and writes into
the repository root:

  * benchmark_report.md - a Markdown report with one table grouped by
    operator (config): the baseline target is normalized to 1.00x and every
    other target shows its speedup relative to the baseline (geometric mean
    over test cases, min-max range in parentheses);
  * benchmark_speedup_bars.png - a bar chart of the per-operator speedups.

Example:
    python3 benchmark_report.py \
        esp32_perf_results.json esp32s3_perf_results.json esp32p4_perf_results.json

The baseline target is auto-detected from the ``target`` field inside each
JSON, so the files may be passed in any order.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Targets whose entries with these statuses must not be used as timings.
BAD_STATUSES = {"error", "failed", "skip", "skipped", "timeout"}

# Matplotlib is imported lazily so that the Markdown output still works
# on machines without it (charts are skipped with a warning).


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _to_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0.0 else None


def load_results(path):
    """Return (meta, name->entry, target) for one perf-results JSON."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    results = doc.get("results", [])
    target = None
    entries = {}
    for e in results:
        name = e.get("name")
        if not name:
            continue
        if target is None and e.get("target"):
            target = str(e["target"])
        status = str(e.get("status", "")).lower()
        if status in BAD_STATUSES:
            continue
        mean = _to_float(e.get("mean_us"))
        if mean is None:  # fall back to median when mean is missing
            mean = _to_float(e.get("median_us"))
        if mean is None:
            continue
        entries[name] = {
            "mean_us": mean,
            "op": str(e.get("config", "")),
        }
        entries[name]["_raw"] = e

    if target is None:  # infer from file name as last resort
        stem = Path(path).name.lower()
        for cand in (
            "esp32p4",
            "esp32s3",
            "esp32s2",
            "esp32c6",
            "esp32c5",
            "esp32c3",
            "esp32c2",
            "esp32",
        ):
            if cand in stem:
                target = cand
                break
        if target is None:
            target = Path(path).stem

    meta = {
        k: doc.get(k)
        for k in (
            "absolute_floor_us",
            "relative_threshold_pct",
            "commit_sha",
            "idf_version",
        )
    }
    return meta, entries, target


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def find_repo_root():
    """Locate the repository root (the dir containing esp-dl/idf_component.yml).

    Works regardless of where this script sits inside the repo; returns None
    if the marker cannot be found.
    """
    marker = Path("esp-dl") / "idf_component.yml"
    p = Path(__file__).resolve().parent
    for parent in (p, *p.parents):
        if (parent / marker).is_file():
            return parent
    return None


def load_espdl_version(component_yml=None):
    """Read the esp-dl version from idf_component.yml (or None if missing)."""
    if component_yml is None:
        repo_root = find_repo_root()
        if repo_root is None:
            return None
        component_yml = repo_root / "esp-dl" / "idf_component.yml"
    path = Path(component_yml)
    if not path.is_file():
        return None
    m = re.search(
        r'^version:\s*["\']?([^"\'\s]+)', path.read_text(encoding="utf-8"), re.MULTILINE
    )
    return m.group(1) if m else None


def fmt_ratio(v, width=2):
    """Format a speedup ratio compactly (3 significant digits)."""
    if not math.isfinite(v):
        return "-"
    if v >= 100:
        s = f"{v:.0f}"
    elif v >= 10:
        s = f"{v:.1f}"
    else:
        s = f"{v:.2f}"
    return s.rjust(width)


# --------------------------------------------------------------------------
# Report building
# --------------------------------------------------------------------------
def build_report(paths, baseline_name):
    metas, targets, data = [], [], []
    for p in paths:
        meta, entries, target = load_results(p)
        metas.append(meta)
        targets.append(target)
        data.append(entries)
        print(f"  loaded {p}: target={target}, {len(entries)} timed entries")

    if len(set(targets)) != len(targets):
        raise SystemExit("error: duplicate target among the input files")

    if baseline_name not in targets:
        raise SystemExit(
            f"error: baseline target '{baseline_name}' not found in inputs "
            f"(targets: {', '.join(targets)})"
        )

    b_idx = targets.index(baseline_name)
    order = [b_idx] + [i for i in range(len(targets)) if i != b_idx]
    targets = [targets[i] for i in order]
    data = [data[i] for i in order]
    metas = [metas[i] for i in order]

    # Test cases common to every target.
    common = set(data[0])
    for entries in data[1:]:
        common &= set(entries)
    common = sorted(common)
    dropped = {t: len(data[i]) - len(common) for i, t in enumerate(targets)}

    # Per-test-case speedups (baseline / target) and per-op aggregation.
    rows = []  # one dict per common test case
    op_stats = {}  # op -> {target: {'gm', 'lo', 'hi'}, 'cases': n}
    for name in common:
        base = data[0][name]
        row = {
            "name": name,
            "op": base["op"],
            "times": {t: data[i][name]["mean_us"] for i, t in enumerate(targets)},
            "speedups": {},
        }
        for i, t in enumerate(targets):
            row["speedups"][t] = base["mean_us"] / data[i][name]["mean_us"]
        rows.append(row)
        op = base["op"]
        st = op_stats.setdefault(op, {"cases": 0, "vals": defaultdict(list)})
        st["cases"] += 1
        for t in targets[1:]:
            st["vals"][t].append(row["speedups"][t])

    for op, st in op_stats.items():
        st["gm"] = {t: geomean(v) for t, v in st["vals"].items()}
        st["lo"] = {t: min(v) for t, v in st["vals"].items()}
        st["hi"] = {t: max(v) for t, v in st["vals"].items()}

    floor = metas[0].get("absolute_floor_us")
    floor_clamped = 0
    for r in rows:
        if floor and any(t <= floor for t in r["times"].values()):
            r["clamped"] = 1
            floor_clamped += 1
        else:
            r["clamped"] = 0

    return {
        "targets": targets,
        "baseline": baseline_name,
        "rows": rows,
        "op_stats": op_stats,
        "floor": floor,
        "floor_clamped": floor_clamped,
        "dropped": dropped,
        "metas": metas,
    }


def write_markdown(report, chart_names, sort_key, version=None):
    targets = report["targets"]
    base = targets[0]
    others = targets[1:]
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    op_stats = report["op_stats"]
    if sort_key == "speedup":
        ops = sorted(
            op_stats, key=lambda o: op_stats[o]["gm"][others[-1]], reverse=True
        )
    else:
        ops = sorted(op_stats)

    lines = []
    lines.append("# ESP-DL Operator Cross-Target Benchmark Report")
    lines.append("")
    lines.append(f"- esp-dl version: **{version or 'unknown'}**")
    lines.append(f"- Generated at: {now}")
    lines.append(f"- Baseline target: **{base}** (normalized to 1.00x)")
    lines.append(f"- Compared targets: {', '.join('**' + t + '**' for t in others)}")
    lines.append(f"- Test cases: {len(report['rows'])} (cases common to every target)")
    lines.append(
        "- Aggregation: per-operator speedup is the **geometric mean** over "
        "its test cases; the range in parentheses is the min-max across "
        "those cases"
    )
    lines.append("")
    lines.append(f"## Per-Operator Speedup ({base} = 1.00x)")
    lines.append("")
    lines.append(f"{len(ops)} operators in total:")
    lines.append("")
    # Table rows follow the same sort order as requested.
    head = "| Operator | Cases | " + " | ".join(targets) + " |"
    lines.append(head)
    lines.append("|---|--:|" + "|".join("---:" for _ in targets) + "|")
    for op in ops:
        st = op_stats[op]
        cells = [f"| {op} | {st['cases']} |", f" 1.00x |"]
        for t in others:
            gm, lo, hi = st["gm"][t], st["lo"][t], st["hi"][t]
            if lo == hi:
                cells.append(f" {fmt_ratio(gm)}x |")
            else:
                cells.append(f" {fmt_ratio(gm)}x ({fmt_ratio(lo)}-{fmt_ratio(hi)}) |")
        lines.append("".join(cells))
    lines.append("")
    for name, title in chart_names:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({name})")
        lines.append("")
    # Sum of mean execution time over all common test cases, per target.
    totals = {t: sum(r["times"][t] for r in report["rows"]) for t in targets}
    lines.append("## Total Test Time")
    lines.append("")
    lines.append(
        "Sum of the mean execution time (us) of all common test " "cases per target:"
    )
    lines.append("")
    lines.append("| Target | Total time (us) |")
    lines.append("|--------|----------------:|")
    for t in targets:
        lines.append(f"| {t} | {totals[t]:,.1f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    notes = [
        f"Speedup = baseline ({base}) time / target time; larger is faster. "
        "A value below 1.00x means the target is slower than the baseline "
        "on that operator."
    ]
    if report["floor"]:
        notes.append(
            f"Raw measurements below {report['floor']} us are clamped in the "
            f"source data; {report['floor_clamped']} test cases are affected, "
            "so their speedups may be overestimated."
        )
    for t, n in report["dropped"].items():
        if n:
            notes.append(
                f"Test cases unique to `{t}` ({n} cases) are excluded from "
                "the comparison; see the raw JSON for that target."
            )
    for i, n in enumerate(notes, 1):
        lines.append(f"{i}. {n}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Charts (matplotlib, optional)
# --------------------------------------------------------------------------
def setup_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            f"warning: matplotlib not available ({e}); charts skipped"
        ) from e

    from matplotlib import font_manager

    # Prefer a CJK-capable font, fall back to DejaVu Sans.
    for fam in (
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ):
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _log_ticks(ax, axis="x"):
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    labels = ["0.25", "0.5", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)


def chart_bars(report, path):
    plt = setup_matplotlib()
    base, others = report["targets"][0], report["targets"][1:]
    op_stats = report["op_stats"]
    sort_t = others[-1]
    ops = sorted(op_stats, key=lambda o: op_stats[o]["gm"][sort_t], reverse=True)

    n_ops, n_grp = len(ops), len(others)
    height = max(6.0, 0.32 * n_ops)
    fig, ax = plt.subplots(figsize=(11, height))
    y = list(range(n_ops))
    colors = [plt.cm.Set2.colors[i % 8] for i in range(n_grp)]

    for gi, t in enumerate(others):
        vals = [op_stats[o]["gm"][t] for o in ops]
        off = (gi - (n_grp - 1) / 2) * 0.36
        yy = [v + off for v in y]
        bars = ax.barh(
            yy,
            vals,
            height=0.34,
            color=colors[gi],
            label=f"{t} (geomean {fmt_ratio(geomean(vals))}x)",
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_width() * 1.02,
                b.get_y() + b.get_height() / 2,
                f"{fmt_ratio(v)}x",
                va="center",
                fontsize=6.5,
                color="#222222",
            )

    ax.axvline(1.0, color="#d62728", lw=1, ls="--")
    ax.text(
        1.0, n_ops - 0.35, " 1.00x (baseline)", color="#d62728", fontsize=8, va="top"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ops, fontsize=7.5)
    ax.set_xscale("log", base=2)
    _log_ticks(ax)
    ax.set_xlabel(f"Speedup vs {base} (log2 scale)")
    ax.set_title(
        f"Per-operator speedup vs {base} (geometric mean, " f"sorted by {sort_t})"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", which="major", alpha=0.3, lw=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a cross-target operator benchmark report "
        "(Markdown + bar chart) from esp-dl perf JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "jsons",
        nargs="+",
        metavar="TARGET_JSON",
        help="per-target perf-results JSON files "
        "(e.g. esp32_perf_results.json ...), any order",
    )
    ap.add_argument(
        "--baseline",
        default="esp32",
        help="baseline target whose performance is normalized " "to 1.00x",
    )
    ap.add_argument(
        "--out-dir", default=None, help="output directory (default: repository root)"
    )
    ap.add_argument(
        "--sort",
        choices=("name", "speedup"),
        default="name",
        help="sort order of the Markdown operator table "
        "(speedup sorts by the last comparison target)",
    )
    ap.add_argument("--no-charts", action="store_true", help="skip chart generation")
    ap.add_argument(
        "--component-yml",
        default=None,
        help="path to esp-dl idf_component.yml used for the "
        "version stamp in the report header "
        "(default: <repo root>/esp-dl/idf_component.yml)",
    )
    args = ap.parse_args(argv)

    for p in args.jsons:
        if not Path(p).is_file():
            raise SystemExit(f"error: file not found: {p}")

    out_dir = Path(args.out_dir) if args.out_dir else find_repo_root() or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== ESP-DL benchmark report ==")
    version = load_espdl_version(args.component_yml)
    print(f"esp-dl version: {version or 'unknown'}")
    report = build_report(args.jsons, args.baseline)
    targets = report["targets"]
    print(
        f"baseline: {targets[0]}, comparison: {', '.join(targets[1:])}, "
        f"common test cases: {len(report['rows'])}"
    )

    # Charts first so the Markdown can reference the files that exist.
    chart_names = []
    if not args.no_charts:
        bar_png = out_dir / "benchmark_speedup_bars.png"
        try:
            chart_bars(report, bar_png)
            chart_names.append(
                (bar_png.name, f"Per-Operator Speedup Bar Chart ({targets[0]} = 1.00x)")
            )
            print(f"  wrote {bar_png}")
        except SystemExit as e:
            print(e)

    md_path = out_dir / "benchmark_report.md"
    md_text = write_markdown(report, chart_names, args.sort, version)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  wrote {md_path}")
    print(f"report dir: {out_dir}")


if __name__ == "__main__":
    main()
