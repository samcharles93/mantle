#!/usr/bin/env python3
"""Summarises benchmark TSVs into a markdown table.

Reads one or more TSVs written by bench_cpu.sh / bench_llamacpp.sh /
bench_cuda.sh and prints median throughput with the observed spread, plus peak
RSS. Container sizes are reported against the bf16 reference so bits per weight
can be compared across formats.
"""

import argparse
import statistics
import sys


def read_tsv(path):
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def aggregate(rows, metrics):
    """Groups rows by label and returns median/min/max for each metric."""
    out = {}
    for row in rows:
        label = row["label"]
        entry = out.setdefault(label, {"rows": []})
        entry["rows"].append(row)
    for label, entry in out.items():
        for metric in metrics:
            values = []
            for row in entry["rows"]:
                raw = row.get(metric, "")
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
            if values:
                entry[metric] = (statistics.median(values), min(values), max(values))
    return out


def fmt(value, digits=2):
    if not value:
        return "-"
    med, lo, hi = value
    if hi - lo < 0.05 * max(med, 1e-9):
        return f"{med:.{digits}f}"
    return f"{med:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--bf16-bytes", type=int, help="reference bf16 container size in bytes")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    rows = []
    for path in args.tsv:
        rows.extend(read_tsv(path))

    metrics = [m for m in ("load_s", "pp_tps", "tg_tps") if any(m in r for r in rows)]
    agg = aggregate(rows, metrics + ["peak_rss_kb", "size_bytes"])

    if args.title:
        print(f"### {args.title}\n")

    header = ["container", "size MiB"]
    if args.bf16_bytes:
        header.append("% of bf16")
    header += ["peak RSS MiB"]
    label_for = {
        "load_s": "load s",
        "pp_tps": "prompt tok/s",
        "tg_tps": "gen tok/s",
    }
    header += [label_for.get(m, m) for m in metrics]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    # Largest container first keeps the baseline at the top of the table.
    def size_of(entry):
        return entry.get("size_bytes", (0, 0, 0))[0]

    for label in sorted(agg, key=lambda k: -size_of(agg[k])):
        entry = agg[label]
        size_med = size_of(entry)
        cells = [label, f"{size_med / (1024 * 1024):.1f}"]
        if args.bf16_bytes:
            cells.append(f"{100.0 * size_med / args.bf16_bytes:.1f}%")
        rss = entry.get("peak_rss_kb")
        cells.append(f"{rss[0] / 1024:.0f}" if rss else "-")
        for m in metrics:
            cells.append(fmt(entry.get(m)))
        print("| " + " | ".join(cells) + " |")

    if any(len(agg[l]["rows"]) > 1 for l in agg):
        print("\nMedian with min-max in brackets where the spread exceeds 5%.")


if __name__ == "__main__":
    sys.exit(main())
