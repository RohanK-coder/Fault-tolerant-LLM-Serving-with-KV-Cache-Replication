#!/usr/bin/env python3
import argparse
import csv
import os
import time

import torch

from common import DEFAULT_MODEL, choose_device, format_prompt, load_model, load_tokenizer, move_batch


def cpu_clone_past(past_key_values):
    return tuple((k.detach().to("cpu").clone(), v.detach().to("cpu").clone()) for k, v in past_key_values)


def move_past_to_device(past_key_values, device: str):
    return tuple((k.to(device), v.to(device)) for k, v in past_key_values)


def pkv_nbytes(past_key_values) -> int:
    total = 0
    for k, v in past_key_values:
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def stepwise_generate(model, input_ids, attention_mask, total_new_tokens):
    generated = input_ids.clone()
    attn = attention_mask.clone()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)

    pkv = out.past_key_values
    snapshots = []

    for step_idx in range(total_new_tokens):
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=attn.device)],
            dim=1,
        )

        snapshots.append(
            {
                "step": step_idx + 1,
                "generated": generated.clone(),
                "attn": attn.clone(),
                "pkv_cpu": cpu_clone_past(pkv),
            }
        )

        with torch.no_grad():
            out = model(
                input_ids=next_id,
                attention_mask=attn,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
        pkv = out.past_key_values

    return generated, attn, snapshots


def continue_from_pkv(model, generated_prefix, attention_mask, pkv, more_tokens):
    generated = generated_prefix.clone()
    attn = attention_mask.clone()

    for _ in range(more_tokens):
        last_id = generated[:, -1:]
        with torch.no_grad():
            out = model(
                input_ids=last_id,
                attention_mask=attn,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=attn.device)],
            dim=1,
        )
        pkv = out.past_key_values

    return generated


def replay_suffix_into_cache(model, base_generated, base_attn, base_pkv, replay_tokens):
    pkv = base_pkv
    attn = base_attn.clone()

    for i in range(replay_tokens.shape[1]):
        tok = replay_tokens[:, i : i + 1]
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=attn,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
        pkv = out.past_key_values
        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=attn.device)],
            dim=1,
        )

    return pkv


def recompute_from_prompt(model, prompt_ids, prompt_attn, total_tokens):
    generated, _, _ = stepwise_generate(model, prompt_ids, prompt_attn, total_tokens)
    return generated


def main():
    parser = argparse.ArgumentParser(description="Part 3: Compare recovery strategies")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why prefix KV replication is a useful recovery strategy."
    )
    parser.add_argument("--generation-tokens", type=int, default=24)
    parser.add_argument("--failure-token", type=int, default=8)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--csv-out", type=str, default="results/part3_strategy_comparison.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    device = choose_device()
    tokenizer = load_tokenizer(args.model)
    model, load_sec = load_model(args.model, device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model loaded in {load_sec:.2f}s")

    prompt = format_prompt(args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = move_batch(inputs, device)

    prompt_ids = inputs["input_ids"]
    prompt_attn = inputs["attention_mask"]

    start = time.time()
    baseline_generated, _, snapshots = stepwise_generate(
        model, prompt_ids, prompt_attn, args.generation_tokens
    )
    baseline_time = time.time() - start

    failure_idx = max(1, min(args.failure_token, len(snapshots))) - 1
    failure_snapshot = snapshots[failure_idx]
    fail_generated = failure_snapshot["generated"]
    fail_attn = failure_snapshot["attn"]
    fail_pkv_cpu = failure_snapshot["pkv_cpu"]
    remaining = args.generation_tokens - (failure_idx + 1)

    rows = []

    # 1) No replication
    t0 = time.time()
    recovered_none = recompute_from_prompt(model, prompt_ids, prompt_attn, args.generation_tokens)
    rec_none = time.time() - t0
    rows.append(
        {
            "strategy": "none",
            "runtime_overhead_sec": 0.0,
            "replicated_kv_mb": 0.0,
            "recovery_time_sec": round(rec_none, 4),
            "matches_baseline": bool(torch.equal(recovered_none, baseline_generated)),
        }
    )

    # 2) Full replication
    full_overhead_start = time.time()
    full_kv_mb = pkv_nbytes(fail_pkv_cpu) / (1024 * 1024)
    full_restore = move_past_to_device(fail_pkv_cpu, device)
    full_overhead = time.time() - full_overhead_start

    t1 = time.time()
    recovered_full = continue_from_pkv(model, fail_generated, fail_attn, full_restore, remaining)
    rec_full = time.time() - t1
    rows.append(
        {
            "strategy": "full",
            "runtime_overhead_sec": round(full_overhead, 4),
            "replicated_kv_mb": round(full_kv_mb, 4),
            "recovery_time_sec": round(rec_full, 4),
            "matches_baseline": bool(torch.equal(recovered_full, baseline_generated)),
        }
    )

    # 3) Selective replication: prompt + recent window checkpoint
    with torch.no_grad():
        prompt_out = model(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            use_cache=True,
            return_dict=True,
        )
    prompt_pkv_cpu = cpu_clone_past(prompt_out.past_key_values)

    # Choose a recent checkpoint before failure
    recent_start_step = max(0, (failure_idx + 1) - args.recent_window)
    if recent_start_step == 0:
        recent_base_generated = prompt_ids
        recent_base_attn = prompt_attn
        recent_base_pkv_cpu = prompt_pkv_cpu
        replay_tokens = fail_generated[:, prompt_ids.shape[1] : -1]
    else:
        recent_snapshot = snapshots[recent_start_step - 1]
        recent_base_generated = recent_snapshot["generated"]
        recent_base_attn = recent_snapshot["attn"]
        recent_base_pkv_cpu = recent_snapshot["pkv_cpu"]
        replay_tokens = fail_generated[:, recent_base_generated.shape[1] - 1 : -1]

    selective_kv_mb = max(
        pkv_nbytes(prompt_pkv_cpu),
        pkv_nbytes(recent_base_pkv_cpu),
    ) / (1024 * 1024)

    select_overhead_start = time.time()
    restored_recent = move_past_to_device(recent_base_pkv_cpu, device)
    select_overhead = time.time() - select_overhead_start

    t2 = time.time()
    rebuilt_pkv = replay_suffix_into_cache(
        model=model,
        base_generated=recent_base_generated,
        base_attn=recent_base_attn,
        base_pkv=restored_recent,
        replay_tokens=replay_tokens,
    )
    recovered_selective = continue_from_pkv(
        model=model,
        generated_prefix=fail_generated,
        attention_mask=fail_attn,
        pkv=rebuilt_pkv,
        more_tokens=remaining,
    )
    rec_selective = time.time() - t2

    rows.append(
        {
            "strategy": f"selective_prefix_plus_recent{args.recent_window}",
            "runtime_overhead_sec": round(select_overhead, 4),
            "replicated_kv_mb": round(selective_kv_mb, 4),
            "recovery_time_sec": round(rec_selective, 4),
            "matches_baseline": bool(torch.equal(recovered_selective, baseline_generated)),
        }
    )

    # 4) Periodic checkpoint baseline for comparison
    periodic_step = max(1, args.recent_window)
    periodic_checkpoint_idx = ((failure_idx + 1) // periodic_step) * periodic_step
    if periodic_checkpoint_idx == (failure_idx + 1):
        periodic_checkpoint_idx = max(0, periodic_checkpoint_idx - periodic_step)

    if periodic_checkpoint_idx == 0:
        periodic_generated = prompt_ids
        periodic_attn = prompt_attn
        periodic_pkv_cpu = prompt_pkv_cpu
        periodic_replay = fail_generated[:, prompt_ids.shape[1] : -1]
    else:
        periodic_snapshot = snapshots[periodic_checkpoint_idx - 1]
        periodic_generated = periodic_snapshot["generated"]
        periodic_attn = periodic_snapshot["attn"]
        periodic_pkv_cpu = periodic_snapshot["pkv_cpu"]
        periodic_replay = fail_generated[:, periodic_generated.shape[1] - 1 : -1]

    periodic_kv_mb = pkv_nbytes(periodic_pkv_cpu) / (1024 * 1024)

    periodic_overhead_start = time.time()
    periodic_restore = move_past_to_device(periodic_pkv_cpu, device)
    periodic_overhead = time.time() - periodic_overhead_start

    t3 = time.time()
    periodic_rebuilt = replay_suffix_into_cache(
        model=model,
        base_generated=periodic_generated,
        base_attn=periodic_attn,
        base_pkv=periodic_restore,
        replay_tokens=periodic_replay,
    )
    recovered_periodic = continue_from_pkv(
        model=model,
        generated_prefix=fail_generated,
        attention_mask=fail_attn,
        pkv=periodic_rebuilt,
        more_tokens=remaining,
    )
    rec_periodic = time.time() - t3

    rows.append(
        {
            "strategy": f"periodic_k{periodic_step}",
            "runtime_overhead_sec": round(periodic_overhead, 4),
            "replicated_kv_mb": round(periodic_kv_mb, 4),
            "recovery_time_sec": round(rec_periodic, 4),
            "matches_baseline": bool(torch.equal(recovered_periodic, baseline_generated)),
        }
    )

    print(f"[INFO] Baseline total generation time: {baseline_time:.4f}s")
    print(f"[INFO] Failure after generated token: {failure_idx + 1}")
    print(f"[INFO] Recent window: {args.recent_window}")
    for row in rows:
        print(row)

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved comparison CSV to: {args.csv_out}")


if __name__ == "__main__":
    main()
