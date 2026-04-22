#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/part1_kv_inspect_tinyllama.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Explain why preserving KV cache helps recover LLM generation after a failure." \
  --max-new-tokens 24 \
  --out results/part1_tinyllama.json
