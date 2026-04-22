#!/usr/bin/env bash
set -e
source .venv/bin/activate

python scripts/part3_compare_strategies.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Explain why selective KV replication with a recent window improves fault tolerant LLM serving." \
  --generation-tokens 24 \
  --failure-token 12 \
  --recent-window 6 \
  --csv-out results/part3_strategy_comparison.csv
