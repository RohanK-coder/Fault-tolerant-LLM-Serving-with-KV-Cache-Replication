#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/part4_plot_results.py \
  --csv results/part3_strategy_comparison.csv \
  --out-dir results/plots
