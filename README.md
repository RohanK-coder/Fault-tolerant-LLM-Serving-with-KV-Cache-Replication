# Selective KV-Cache Replication for Fault-Tolerant LLM Serving

A **terminal-first** software prototype and architecture study of **KV-cache checkpointing and selective replication** for resilient autoregressive LLM inference, built for **advanced computer architecture / systems-oriented ML** coursework (adapted for **Apple Silicon** laptops).

This project treats fault-tolerant decoding as more than a modeling exercise. It studies the problem as a **memory-system and control tradeoff**:

- How does the **KV cache** grow with prompt and generated length, and how can we **measure** it layer by layer?
- After a simulated failure mid-generation, how do we **resume** from a saved prompt KV state without silently drifting from a full reroll baseline?
- How do **recovery strategies** compare—**no replication**, **full snapshot replication**, **selective (prefix + recent-window) replication**, and a **periodic** checkpoint baseline?
- What do **recovery time**, **replicated KV volume (MB)**, **runtime overhead of restore**, and **correctness (token match)** imply for real serving stacks?

---

## 1. Project Summary

In baseline greedy decoding, the model maintains **past key–values (KV)** for all prior tokens. After a failure, a naïve recovery may **recompute from the prompt**, which is correct but expensive. Alternatively, the system can **replicate** portions of the KV state at runtime so recovery can **continue from a checkpoint** with less rework.

This repository implements and compares several strategies on a **single target model** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`): there is no separate draft model; the emphasis is on **what to replicate**, **when**, and **how much memory** that costs versus **recovery latency**.

The workflow is organized into **four parts** you run locally in order (shell wrappers + Python scripts), not on a fixed weekly schedule.

---

## 2. Main Contributions

- **Part 1 — KV inspection:** layer-wise KV shapes, dtypes, and total KV memory for a short greedy generation; JSON summary for reports.
- **Part 2 — Prompt checkpoint + resume:** save CPU-cloned prompt `past_key_values` (and associated logits), simulate generate-then-fail-then-recover, and verify **exact token match** against a full baseline run.
- **Part 3 — Strategy comparison:** CSV metrics for `none`, `full`, selective prefix + recent window, and periodic-K checkpoint strategies (`recovery_time_sec`, `replicated_kv_mb`, `runtime_overhead_sec`, `matches_baseline`).
- **Part 4 — Plotting:** bar and scatter plots from the Part 3 comparison CSV (recovery time, replication cost, cost–recovery tradeoff).
- **Optional multi-trial pipeline:** repeat Part 3 many times, aggregate means/stds, and plot summary figures.

---

## 3. Repository Layout

> Paths below match this repository. If you rename directories locally, adjust commands accordingly.

```text
tinyllama_kv_project/
├── scripts/
│   ├── common.py
│   ├── part1_kv_inspect_tinyllama.py
│   ├── part2_save_resume.py
│   ├── part3_compare_strategies.py
│   ├── part4_plot_results.py
│   ├── run_part3_trials.py
│   ├── aggregate_part3_trials.py
│   └── plot_part3_summary.py
├── results/                 # generated; see .gitignore
│   ├── part1_tinyllama.json
│   ├── part2_resume/
│   ├── part3_strategy_comparison.csv
│   ├── plots/
│   └── ...
├── logs/
├── requirements.txt
├── .gitignore
├── run_part1_tinyllama.sh
├── run_part2_resume.sh
├── run_part3_compare.sh
├── run_part4_plot.sh
└── run_part3_trials_and_summary.sh
```

### Important files

| File | Role |
|------|------|
| `scripts/common.py` | Device selection (MPS vs CPU), model/tokenizer load, prompt formatting, JSON helpers. |
| `scripts/part1_kv_inspect_tinyllama.py` | Inspect KV structure and memory; write `results/part1_tinyllama.json` (default). |
| `scripts/part2_save_resume.py` | Save prompt KV, benchmark recovery path, write summary under `results/part2_resume/` (default). |
| `scripts/part3_compare_strategies.py` | Run strategy comparison; write CSV (default `results/part3_strategy_comparison.csv`). |
| `scripts/part4_plot_results.py` | Read comparison CSV; write PNGs under `results/plots/` (default). |
| `scripts/run_part3_trials.py` | Subprocess driver: many trials → `results/part3_trials.csv`. |
| `scripts/aggregate_part3_trials.py` | Aggregate trials → `results/part3_summary.csv`. |
| `scripts/plot_part3_summary.py` | Plot aggregated metrics → `results/plots_summary/`. |

---

## 4. Supported Model

The project is configured around **one** causal LM (easy on laptops, still structurally representative):

```text
DEFAULT_MODEL = TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

You can override `--model` on each script if you swap in another Hugging Face causal LM, but **layer counts, KV layout, and memory** will change; re-validate correctness on Part 2 before trusting Part 3 comparisons.

---

## 5. Environment Setup

### Recommended Python version

Use **Python 3.9–3.12** for broad `torch` / `transformers` compatibility. Newer Python versions may work but are not guaranteed against every wheel.

### Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Core packages (see `requirements.txt`): `torch`, `transformers`, `accelerate`, `sentencepiece`, `matplotlib`, `pandas`.

### Notes

- First run will **download** the TinyLlama weights from Hugging Face (network + disk).
- On **Apple Silicon**, `scripts/common.py` uses **MPS** when available; otherwise **CPU**.
- Hugging Face cache defaults to your user cache; a local `.cache/` directory may appear and is listed in `.gitignore`.

---

## 6. Running the Project

### Part 1 — KV inspection

```bash
./run_part1_tinyllama.sh
```

Or directly:

```bash
python scripts/part1_kv_inspect_tinyllama.py --help
```

### Part 2 — Save / resume from prompt KV

```bash
./run_part2_resume.sh
```

### Part 3 — Single comparison (writes CSV)

```bash
./run_part3_compare.sh
```

### Part 4 — Plots from Part 3 CSV

```bash
./run_part4_plot.sh
```

Requires Part 3 output (default path: `results/part3_strategy_comparison.csv`).

### Optional — Multi-trial Part 3 + aggregation + plots

```bash
./run_part3_trials_and_summary.sh
```

This runs multiple trials, aggregates by strategy, and writes figures under `results/plots_summary/`.

---

## 7. Reproducibility Guide

### Step 1 — Verify the environment

```bash
python -c "import torch, transformers, matplotlib, pandas; print('env ok', torch.backends.mps.is_available())"
```

### Step 2 — Run parts in order

1. Part 1 → JSON KV report  
2. Part 2 → checkpoint + resume summary  
3. Part 3 → strategy comparison CSV  
4. Part 4 → plots from that CSV  

### Step 3 — Validate correctness (Part 2 / Part 3)

- Part 2 prints whether recovered tokens **exactly match** the baseline path.
- Part 3 CSV includes `matches_baseline` per strategy; selective and periodic paths should match when the implementation assumptions hold.

### Step 4 — Sweeps (manual)

`part3_compare_strategies.py` accepts at least:

- `--generation-tokens`, `--failure-token`, `--recent-window`, `--prompt`, `--csv-out`

Example:

```bash
python scripts/part3_compare_strategies.py \
  --generation-tokens 32 \
  --failure-token 10 \
  --recent-window 8 \
  --csv-out results/part3_strategy_comparison.csv
```

For repeated runs, use `run_part3_trials.py` and then aggregation/plot scripts.

### Step 5 — Artifacts directory

By default, artifacts go under `results/` (ignored by git in this repo—regenerate after clone). Keep CSVs and plots together for reports:

- `results/part3_strategy_comparison.csv`
- `results/plots/*.png`
- `results/plots_summary/*.png` (multi-trial path)

---

## 8. Important Note for Reproducibility

- The **code path is reproducible**: same flags and hardware class should yield consistent **trends**; absolute wall-clock times vary by machine and load.
- **Git clones** that respect `.gitignore` will **not** include `results/`; rerun the shell scripts to regenerate figures and tables.
- For strict apples-to-apples numbers for a paper, **record** `torch`/`transformers` versions, device (`mps` vs `cpu`), and fixed seeds if you extend the code with sampling.

---

## 9. How to Generate the Plots

### Option A — From a single Part 3 run

1. Run `./run_part3_compare.sh` (or `part3_compare_strategies.py` with your flags).  
2. Run `./run_part4_plot.sh`.

Outputs (defaults):

```text
results/plots/recovery_time_by_strategy.png
results/plots/replicated_kv_by_strategy.png
results/plots/cost_vs_recovery_tradeoff.png
```

### Option B — From multi-trial data

1. `./run_part3_trials_and_summary.sh`  
2. Inspect `results/part3_summary.csv` and `results/plots_summary/*.png`.

---

## 10. Recommended Plot / Metric Set

Useful for slides and reports:

1. **Recovery time by strategy** — primary latency comparison.  
2. **Replicated KV (MB) by strategy** — memory/replication cost.  
3. **Scatter: replicated KV vs recovery time** — cost–latency tradeoff.  
4. **Multi-trial means** (optional) — stability across prompts/trials via `part3_summary.csv`.

---

## 11. Suggested Reproduction Commands

**Minimal end-to-end (after `pip install` and venv activate):**

```bash
./run_part1_tinyllama.sh
./run_part2_resume.sh
./run_part3_compare.sh
./run_part4_plot.sh
```

**Heavier multi-trial pipeline:**

```bash
./run_part3_trials_and_summary.sh
```

---

## 12. Expected Findings

Typical patterns (your exact numbers depend on hardware and hyperparameters):

- **No replication** (`none`): higher recovery time (recompute-heavy) but **no** replicated KV at failure time in the measured sense used for that row.  
- **Full replication** (`full`): lower recovery time after failure but **larger** replicated KV and snapshot overhead.  
- **Selective** strategies: attempt to sit **between** those extremes—lower replication bulk than full snapshot for comparable recovery behavior when assumptions match.  
- **Periodic** baselines: useful comparison when checkpoints are evenly spaced.

One sentence you can defend in a report:

> **Selective prompt KV replication** can reduce recovery cost versus **no replication**, while avoiding the **highest** runtime/memory overhead of **replicating the entire generated KV state**.

---

## 13. Limitations

- **Single-model** study: no draft/target speculative pair; focus is KV recovery, not speculative decoding.  
- **Greedy** decoding only in the provided scripts; sampling would need extension.  
- **Simulated** failure point via stepwise generation snapshots—not a distributed crash or kernel failure.  
- **TinyLlama** fits laptops; absolute KV sizes differ on 7B/70B-class models, though **relative** tradeoffs often track qualitatively.  
- **MPS vs CPU** performance characteristics differ from CUDA servers.

---

## 14. Future Work

- Multi-GPU / vLLM-style serving integration and real request-level failures.  
- Quantized KV (FP8/INT8) and its effect on replication cost.  
- Async replication and bounded staleness policies.  
- Larger models and families with gated checkpoints.  
- Energy measurement tied to KV movement, not only wall-clock.

---

## 15. References / Starting Points

Useful background (not exhaustive):

- Hugging Face **Transformers** (`past_key_values`, `use_cache`, generation).  
- **vLLM** / **PagedAttention** (production KV memory layout).  
- **llama.cpp** and other runtimes (KV cache in edge deployment).  
- PyTorch **MPS** documentation for Apple Silicon.

---

## 16. Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
./run_part1_tinyllama.sh
```

Then run Parts 2–4 in order (or jump to Part 3–4 if you only need strategy plots and regenerate CSVs first).

---

## 17. Reproducibility Statement

This repository is **functionally reproducible**:

- Each part can be rerun from a clean `results/` directory.  
- CSV and PNG outputs are **derived artifacts**; this repo’s `.gitignore` excludes `results/` so clones stay small—**regenerate** outputs after clone for identical figures to your machine’s run.

---

## 18. Authors

Update for your course roster:

- *(Add names here for your submission.)*
# Fault-tolerant-LLM-Serving-with-KV-Cache-Replication
