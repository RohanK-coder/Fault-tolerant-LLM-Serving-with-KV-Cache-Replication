#!/usr/bin/env bash
set -e
source .venv/bin/activate

python scripts/run_part3_trials.py \
  --trials 5 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Explain in detail how fault-tolerant LLM serving works, why KV cache matters, how selective KV replication differs from full replication, and why long prompts and later failures make recovery more expensive." \
  --generation-tokens 40 \
  --failure-token 20 \
  --recent-window 8 \
  --out-csv results/part3_trials.csv

python scripts/aggregate_part3_trials.py \
  --in-csv results/part3_trials.csv \
  --out-csv results/part3_summary.csv

python scripts/plot_part3_summary.py \
  --csv results/part3_summary.csv \
  --out-dir results/plots_summary
