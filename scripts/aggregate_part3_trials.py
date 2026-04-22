#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def stddev(vals):
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def to_float(row, key):
    return float(row[key])


def to_bool(row, key):
    return str(row[key]).strip().lower() == "true"


def main():
    parser = argparse.ArgumentParser(description="Aggregate Part 3 multi-trial results.")
    parser.add_argument("--in-csv", type=str, default="results/part3_trials.csv")
    parser.add_argument("--out-csv", type=str, default="results/part3_summary.csv")
    args = parser.parse_args()

    grouped = defaultdict(list)

    with open(args.in_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["strategy"]].append(row)

    summary_rows = []
    for strategy, rows in grouped.items():
        recovery = [to_float(r, "recovery_time_sec") for r in rows]
        kv = [to_float(r, "replicated_kv_mb") for r in rows]
        overhead = [to_float(r, "runtime_overhead_sec") for r in rows]
        matches = [to_bool(r, "matches_baseline") for r in rows]

        summary_rows.append({
            "strategy": strategy,
            "num_trials": len(rows),
            "mean_recovery_time_sec": round(mean(recovery), 4),
            "std_recovery_time_sec": round(stddev(recovery), 4),
            "mean_replicated_kv_mb": round(mean(kv), 4),
            "std_replicated_kv_mb": round(stddev(kv), 4),
            "mean_runtime_overhead_sec": round(mean(overhead), 4),
            "all_match_baseline": all(matches),
        })

    summary_rows.sort(key=lambda r: r["mean_recovery_time_sec"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "num_trials",
                "mean_recovery_time_sec",
                "std_recovery_time_sec",
                "mean_replicated_kv_mb",
                "std_replicated_kv_mb",
                "mean_runtime_overhead_sec",
                "all_match_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[INFO] Wrote summary to: {args.out_csv}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
