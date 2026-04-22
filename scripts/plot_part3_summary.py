#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated Part 3 results.")
    parser.add_argument("--csv", type=str, default="results/part3_summary.csv")
    parser.add_argument("--out-dir", type=str, default="results/plots_summary")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    plt.figure(figsize=(9, 5))
    plt.bar(df["strategy"], df["mean_recovery_time_sec"])
    plt.ylabel("Mean recovery time (sec)")
    plt.title("Mean Recovery Time by Strategy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "mean_recovery_time_by_strategy.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(df["strategy"], df["mean_replicated_kv_mb"])
    plt.ylabel("Mean replicated KV (MB)")
    plt.title("Mean Replicated KV by Strategy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "mean_replicated_kv_by_strategy.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(df["mean_replicated_kv_mb"], df["mean_recovery_time_sec"])
    for _, row in df.iterrows():
        plt.annotate(row["strategy"], (row["mean_replicated_kv_mb"], row["mean_recovery_time_sec"]))
    plt.xlabel("Mean replicated KV (MB)")
    plt.ylabel("Mean recovery time (sec)")
    plt.title("Mean Cost vs Recovery Tradeoff")
    plt.tight_layout()
    out3 = os.path.join(args.out_dir, "mean_cost_vs_recovery_tradeoff.png")
    plt.savefig(out3, dpi=200)
    plt.close()

    print(f"[INFO] Saved plots to: {args.out_dir}")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
