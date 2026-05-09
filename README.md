#CECS 574 - Topics Distributed Computing - TEAM DND
# Selective KV-Cache Replication for Fault-Tolerant LLM Serving

A software prototype and architecture study of **KV-cache checkpointing and selective replication** for resilient autoregressive LLM inference, built for **CECS 574 - Topics Distributed Computing**.

This project studies fault-tolerant LLM decoding as more than a modeling exercise. It treats the problem as a **memory-system, checkpointing, and recovery-control tradeoff**: prompt KV caching, generated-token KV growth, failure simulation, checkpoint restore, correctness validation, and selective replication are modeled as interacting parts of a fault-tolerant inference pipeline.

Results generated are also included in the repo.
---

## Reproduction Note

This repository uses a **single main workflow**. There are no separate Git branches for reduced or full experiments.

The project is organized into four local execution parts:

1. **Part 1 — KV inspection**
2. **Part 2 — Prompt checkpoint + resume**
3. **Part 3 — Recovery strategy comparison**
4. **Part 4 — Plot generation**

Generated CSV, JSON, and PNG outputs are written under `results/`. The `results/` directory may be ignored by Git, so outputs can be regenerated locally by rerunning the provided shell scripts.

For quick verification, run Parts 1–4 in order.

---

## Quick Start

Use this path first when checking the project locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

./run_part1_tinyllama.sh
./run_part2_resume.sh
./run_part3_compare.sh
./run_part4_plot.sh
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\part1_kv_inspect_tinyllama.py
python scripts\part2_save_resume.py
python scripts\part3_compare_strategies.py
python scripts\part4_plot_results.py
```

---

## 1. Project Summary

In baseline autoregressive decoding, a causal language model generates one token at a time. To avoid recomputing attention over all previous tokens, the model stores **past key-values**, commonly called the **KV cache**.

During long generation, the KV cache grows with:

- prompt length,
- generated length,
- number of transformer layers,
- number of attention heads / KV heads,
- hidden dimension,
- and tensor dtype.

If an inference worker fails mid-generation, a naive recovery strategy may recompute from the original prompt. That is correct, but expensive. A fault-tolerant serving system can instead replicate selected KV-cache states so recovery can resume from a checkpoint with less rework.

This repository studies KV-cache recovery from three perspectives:

- **Correctness** — recovered generation should exactly match a full baseline greedy decoding run.
- **Memory-system cost** — replicated KV state consumes memory and transfer bandwidth.
- **Recovery tradeoff** — different checkpointing strategies reduce recovery latency at different replication costs.

Key architecture questions explored in this project include:

- How does the KV cache grow across layers during autoregressive generation?
- How much KV state must be replicated to recover efficiently?
- Can selective replication reduce recovery cost without the full memory overhead of complete snapshot replication?
- How do recovery time, replicated KV volume, restore overhead, and token correctness compare across strategies?

---

## 2. Main Contributions

This project includes:

- a **KV-cache inspection tool** for TinyLlama,
- layer-wise reporting of KV tensor shapes, dtypes, and memory size,
- prompt KV checkpointing using CPU-cloned `past_key_values`,
- a generate-then-fail-then-recover simulation,
- exact token-match validation against a full greedy decoding baseline,
- multiple recovery strategies:
  - no replication,
  - full snapshot replication,
  - selective prefix + recent-window replication,
  - periodic checkpoint replication,
- CSV metrics for recovery experiments,
- plot scripts for report and presentation figures,
- optional multi-trial execution and aggregation,
- an Apple Silicon-friendly execution path using PyTorch MPS when available.

The main metrics studied are:

- `recovery_time_sec`,
- `replicated_kv_mb`,
- `runtime_overhead_sec`,
- `matches_baseline`.

---

## 3. Repository Layout

The root-level project structure is:

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
├── results/
│   ├── part1_tinyllama.json
│   ├── part2_resume/
│   ├── part3_strategy_comparison.csv
│   ├── plots/
│   └── plots_summary/
├── logs/
├── requirements.txt
├── .gitignore
├── README.md
├── run_part1_tinyllama.sh
├── run_part2_resume.sh
├── run_part3_compare.sh
├── run_part4_plot.sh
└── run_part3_trials_and_summary.sh
```

> Note: A local `.venv/` directory may exist on the developer machine, but it should be recreated locally by each user and should not be committed.

### Important files and directories

- `scripts/common.py`  
  Shared helpers for device selection, model/tokenizer loading, prompt formatting, and JSON output.

- `scripts/part1_kv_inspect_tinyllama.py`  
  Runs a short greedy generation and records KV-cache structure, tensor shapes, dtype information, and total memory usage.

- `scripts/part2_save_resume.py`  
  Saves prompt KV state, simulates failure during generation, restores from checkpoint, and verifies recovered tokens against a baseline run.

- `scripts/part3_compare_strategies.py`  
  Compares recovery strategies and writes metrics to `results/part3_strategy_comparison.csv`.

- `scripts/part4_plot_results.py`  
  Reads the Part 3 CSV and generates plots under `results/plots/`.

- `scripts/run_part3_trials.py`  
  Runs Part 3 repeatedly and writes multi-trial results.

- `scripts/aggregate_part3_trials.py`  
  Aggregates trial results into summary statistics.

- `scripts/plot_part3_summary.py`  
  Generates plots from aggregated multi-trial results.

- `results/`  
  Generated experiment artifacts such as CSV, JSON, and PNG files.

- `logs/`  
  Optional runtime logs.

---

## 4. Supported Model

The project is configured around this language model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The default model is defined as:

```python
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```


You can override the model with `--model` on supported scripts, but changing the model changes:

- number of layers,
- KV tensor layout,
- KV memory size,
- runtime behavior,
- and recovery performance.

If you use another Hugging Face causal LM, rerun Part 2 and verify exact token-match correctness before trusting Part 3 comparisons.

---

## 5. Environment Setup

### Recommended Python version

Use **Python 3.9-3.12**.

Newer Python versions may work, but package compatibility for `torch`, `transformers`, and Apple Silicon MPS support may vary.

### Create and activate a virtual environment

Create a local virtual environment so dependencies are isolated from system-level Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Core packages include:

- `torch`,
- `transformers`,
- `accelerate`,
- `sentencepiece`,
- `matplotlib`,
- `pandas`.

### Verify the environment

```bash
python -c "import torch, transformers, matplotlib, pandas; print('env ok', torch.backends.mps.is_available())"
```

### Environment notes

- Run all commands from the project root.
- Activate `.venv` before running scripts.
- The first run downloads TinyLlama weights from Hugging Face.
- On Apple Silicon, the code uses **MPS** when available.
- If MPS is unavailable, the code falls back to CPU.
- A local Hugging Face cache may be created automatically.
- The `results/` directory contains generated artifacts and can be regenerated.

---

## 6. Reproducibility Guide

This section gives a complete workflow for inspecting KV cache behavior, validating recovery correctness, comparing strategies, and regenerating plots.

The repository is designed to be **functionally reproducible**. The same scripts and flags should reproduce the same workflow, though absolute wall-clock times may vary by machine, load, device, and package versions.

### Step 1 — Activate the virtual environment

```bash
source .venv/bin/activate
```

If the virtual environment does not exist yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2 — Run KV-cache inspection

```bash
./run_part1_tinyllama.sh
```

Or directly:

```bash
python scripts/part1_kv_inspect_tinyllama.py
```

Default output:

```text
results/part1_tinyllama.json
```

This step records layer-wise KV-cache information and total KV memory usage.

### Step 3 — Run prompt checkpoint and resume validation

```bash
./run_part2_resume.sh
```

Or directly:

```bash
python scripts/part2_save_resume.py
```

Default output directory:

```text
results/part2_resume/
```

This step verifies whether recovered generation exactly matches the full baseline greedy decoding path.

### Step 4 — Run recovery strategy comparison

```bash
./run_part3_compare.sh
```

Or directly:

```bash
python scripts/part3_compare_strategies.py
```

Default output:

```text
results/part3_strategy_comparison.csv
```

This step compares:

- no replication,
- full replication,
- selective prefix + recent-window replication,
- periodic checkpointing.

### Step 5 — Generate plots

```bash
./run_part4_plot.sh
```

Or directly:

```bash
python scripts/part4_plot_results.py
```

Default plot outputs:

```text
results/plots/recovery_time_by_strategy.png
results/plots/replicated_kv_by_strategy.png
results/plots/cost_vs_recovery_tradeoff.png
```

### Step 6 — Optional multi-trial workflow

For repeated trials, aggregation, and summary plots:

```bash
./run_part3_trials_and_summary.sh
```

Typical outputs:

```text
results/part3_trials.csv
results/part3_summary.csv
results/plots_summary/
```

---

## 7. Included / Expected Outputs and Plot Set

The generated CSV / JSON outputs include fields such as:

- strategy name,
- generation token count,
- failure token position,
- recent-window size,
- recovery time,
- replicated KV size in MB,
- restore/runtime overhead,
- baseline token sequence,
- recovered token sequence,
- exact token-match result.

### Main output files

```text
results/part1_tinyllama.json
results/part2_resume/
results/part3_strategy_comparison.csv
results/part3_trials.csv
results/part3_summary.csv
```

### Recommended / generated plot set

1. **Recovery time by strategy**  
   Shows which strategy recovers fastest after a simulated failure.

2. **Replicated KV by strategy**  
   Shows memory cost of each replication policy.

3. **Cost vs recovery tradeoff**  
   Shows the relationship between replicated KV size and recovery latency.

4. **Multi-trial summary plots**  
   Shows mean and variation across repeated Part 3 runs.

---

## 8. Suggested Commands

### KV inspection

```bash
source .venv/bin/activate
./run_part1_tinyllama.sh
```

### Prompt checkpoint + resume

```bash
source .venv/bin/activate
./run_part2_resume.sh
```

### Strategy comparison

```bash
source .venv/bin/activate
./run_part3_compare.sh
```

### Plot generation

```bash
source .venv/bin/activate
./run_part4_plot.sh
```

### Multi-trial experiment

```bash
source .venv/bin/activate
./run_part3_trials_and_summary.sh
```

### Manual Part 3 sweep example

```bash
python scripts/part3_compare_strategies.py \
  --generation-tokens 32 \
  --failure-token 10 \
  --recent-window 8 \
  --csv-out results/part3_strategy_comparison.csv
```

---

## 9. Expected Findings

Across the project, the main expected conclusions are:

- **No replication** has the lowest replication cost but higher recovery latency because more work must be recomputed.
- **Full replication** reduces recovery latency but has the highest replicated KV memory cost.
- **Selective replication** attempts to preserve the most useful KV state while avoiding full snapshot cost.
- **Periodic checkpointing** provides a structured baseline where recovery cost depends on checkpoint interval and failure position.
- Correct recovery requires exact KV/logit state handling; otherwise, resumed generation can silently drift from the baseline.

In one sentence:

> Selective prompt KV replication can reduce recovery cost compared with no replication while avoiding the highest runtime and memory overhead of replicating the entire generated KV state.

---

## 10. Results Summary

Representative results should be generated locally by running Part 3.

A typical comparison table has the following structure:

| Strategy | Recovery Time (sec) | Replicated KV (MB) | Runtime Overhead (sec) | Matches Baseline |
|---|---:|---:|---:|---|
| none | generated locally | generated locally | generated locally | true / false |
| full | generated locally | generated locally | generated locally | true / false |
| selective | generated locally | generated locally | generated locally | true / false |
| periodic | generated locally | generated locally | generated locally | true / false |

Use this CSV as the source of truth:

```text
results/part3_strategy_comparison.csv
```

For multi-trial summaries, use:

```text
results/part3_summary.csv
```

---

## 11. Limitations

- The project uses a **single target model** rather than a full production model fleet.
- The provided scripts use **greedy decoding** only.
- Failures are **simulated** at controlled token positions.
- The system does not implement a real distributed crash, scheduler failure, or GPU worker restart.
- TinyLlama is suitable for laptops, but absolute KV-cache sizes differ from 7B, 13B, or 70B models.
- Apple Silicon MPS behavior differs from CUDA server behavior.
- Wall-clock timings vary across hardware, package versions, system load, and device backend.
- Selective checkpointing policies are prototype policies, not production serving policies.

---

## 12. Troubleshooting

### `ModuleNotFoundError`

Make sure you are running commands from the project root and that the virtual environment is activated.

```bash
source .venv/bin/activate
python scripts/part1_kv_inspect_tinyllama.py
```

### `python3: command not found`

Check the Python version installed on your system:

```bash
python --version
python3 --version
```

Then create the virtual environment using the available command:

```bash
python -m venv .venv
```

or:

```bash
python3 -m venv .venv
```

### `.venv/bin/activate: no such file or directory`

The virtual environment has not been created yet.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Hugging Face model download issues

The first run downloads TinyLlama from Hugging Face. Make sure you have network access and enough disk space.

If the model cache becomes corrupted, remove the local cache entry and rerun the script.

### MPS is unavailable

If this check prints `False`:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

the scripts should fall back to CPU. CPU runs are slower but still useful for correctness testing.

### Plot script cannot find CSV files

Run Part 3 before Part 4:

```bash
./run_part3_compare.sh
./run_part4_plot.sh
```

Also confirm that the CSV exists:

```bash
ls results/part3_strategy_comparison.csv
```

---

## 13. Reproducibility Statement

This repository is designed to be **functionally reproducible**:

- KV-cache inspection can be rerun,
- checkpoint and resume behavior can be revalidated,
- recovery strategies can be compared from the same scripts,
- CSV outputs can be regenerated,
- plots can be rebuilt from CSV data,
- and multi-trial summaries can be reproduced locally.

For strict apples-to-apples reporting, record:

- Python version,
- `torch` version,
- `transformers` version,
- device backend (`mps` or `cpu`),
- model name,
- generation token count,
- failure token position,
- recent-window size,
- and prompt text.

---

## 14. Future Work

Possible extensions include:

- real distributed serving integration,
- vLLM-style KV-page checkpointing,
- async KV replication,
- bounded-staleness checkpoint policies,
- quantized KV replication using FP8 or INT8,
- larger model experiments,
- CUDA server benchmarking,
- energy measurement tied to KV movement,
- request-level failure injection,
- and integration with production inference runtimes.

---

## 15. References / Starting Points

Useful public systems and background topics related to this project include:

- Hugging Face Transformers documentation for `past_key_values`, `use_cache`, and generation.
- PyTorch MPS documentation for Apple Silicon execution.
- vLLM and PagedAttention for production KV-cache memory management.
- llama.cpp and other local inference runtimes for edge deployment.
- Transformer attention and KV-cache design in autoregressive decoding systems.

These references are useful for understanding:

- autoregressive decoding,
- KV-cache growth,
- memory movement,
- checkpointing,
- restore overhead,
- inference serving failures,
- and fault-tolerant LLM runtime design.

---

## 16. Authors

- Yanni Rohan Kommathoti  
  `YanniRohan.Kommathoti01@student.csulb.edu`

- Nikhil Peravali  
  `Nikhil.Peravali01@student.csulb.edu`

- Rajiv Sai Charan Tirumalasetti  
  `RajivSaiCharan.Tirumalasetti01@student.csulb.edu`

---

## Course

**CECS 574 - Topics Distributed Computing**

California State University, Long Beach
