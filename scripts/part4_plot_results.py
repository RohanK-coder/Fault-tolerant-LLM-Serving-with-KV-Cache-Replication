#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Part 4: Plot strategy comparison results")
    parser.add_argument("--csv", type=str, default="results/part3_strategy_comparison.csv")
    parser.add_argument("--out-dir", type=str, default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    label_map = {
        "none": "none",
        "full": "full",
        "selective_prefix_plus_recent6": "selective",
        "periodic_k6": "periodic",
    }
    df["display_strategy"] = df["strategy"].map(lambda s: label_map.get(s, s))

    plt.figure(figsize=(8, 5))
    plt.bar(df["display_strategy"], df["recovery_time_sec"])
    plt.ylabel("Recovery time (sec)")
    plt.title("Recovery Time by Strategy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "recovery_time_by_strategy.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(df["display_strategy"], df["replicated_kv_mb"])
    plt.ylabel("Replicated KV (MB)")
    plt.title("Replication Cost by Strategy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "replicated_kv_by_strategy.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["replicated_kv_mb"], df["recovery_time_sec"])
    for _, row in df.iterrows():
        plt.annotate(row["display_strategy"], (row["replicated_kv_mb"], row["recovery_time_sec"]))
    plt.xlabel("Replicated KV (MB)")
    plt.ylabel("Recovery time (sec)")
    plt.title("Cost vs Recovery Tradeoff")
    plt.tight_layout()
    out3 = os.path.join(args.out_dir, "cost_vs_recovery_tradeoff.png")
    plt.savefig(out3, dpi=200)
    plt.close()

    print(f"Saved plots to: {args.out_dir}")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
