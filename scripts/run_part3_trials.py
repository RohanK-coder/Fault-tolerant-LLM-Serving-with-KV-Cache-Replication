#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import tempfile


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Run Part 3 multiple times and collect all trial rows.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in detail how fault-tolerant LLM serving works, why KV cache matters, how selective KV replication differs from full replication, and why long prompts and later failures make recovery more expensive.",
    )
    parser.add_argument("--generation-tokens", type=int, default=40)
    parser.add_argument("--failure-token", type=int, default=20)
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--out-csv", type=str, default="results/part3_trials.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    all_rows = []

    for trial in range(1, args.trials + 1):
        fd, tmp_csv = tempfile.mkstemp(suffix=".csv")
        os.close(fd)

        cmd = [
            sys.executable,
            "scripts/part3_compare_strategies.py",
            "--model", args.model,
            "--prompt", args.prompt,
            "--generation-tokens", str(args.generation_tokens),
            "--failure-token", str(args.failure_token),
            "--recent-window", str(args.recent_window),
            "--csv-out", tmp_csv,
        ]

        print(f"[RUN] trial={trial}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        rows = read_csv_rows(tmp_csv)
        os.remove(tmp_csv)

        for row in rows:
            row["trial"] = trial
            all_rows.append(row)

    fieldnames = [
        "trial",
        "strategy",
        "runtime_overhead_sec",
        "replicated_kv_mb",
        "recovery_time_sec",
        "matches_baseline",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[INFO] Wrote trial results to: {args.out_csv}")


if __name__ == "__main__":
    main()
