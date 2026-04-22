#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/part2_save_resume.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Explain how saving prompt KV speeds up recovery after a generation failure." \
  --tokens-before-checkpoint 8 \
  --tokens-after-checkpoint 12 \
  --out-dir results/part2_resume
