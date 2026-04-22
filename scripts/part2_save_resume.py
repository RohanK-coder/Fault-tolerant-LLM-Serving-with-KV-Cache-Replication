#!/usr/bin/env python3
import argparse
import os
import time

import torch

from common import (
    DEFAULT_MODEL,
    choose_device,
    format_prompt,
    load_model,
    load_tokenizer,
    move_batch,
    save_json,
)


def cpu_clone_past(past_key_values):
    return tuple((k.detach().to("cpu").clone(), v.detach().to("cpu").clone()) for k, v in past_key_values)


def move_past_to_device(past_key_values, device: str):
    return tuple((k.to(device), v.to(device)) for k, v in past_key_values)


def greedy_generate_from_state(model, generated, attn, pkv, logits, num_steps):
    """
    Continue greedy generation from an existing model state.
    - pkv corresponds to `generated`
    - logits are the logits produced after processing `generated`
    """
    current_generated = generated.clone()
    current_attn = attn.clone()
    current_pkv = pkv
    current_logits = logits

    for _ in range(num_steps):
        next_id = torch.argmax(current_logits[:, -1, :], dim=-1, keepdim=True)

        current_generated = torch.cat([current_generated, next_id], dim=1)
        current_attn = torch.cat(
            [current_attn, torch.ones((current_attn.shape[0], 1), dtype=current_attn.dtype, device=current_attn.device)],
            dim=1,
        )

        with torch.no_grad():
            out = model(
                input_ids=next_id,
                attention_mask=current_attn,
                past_key_values=current_pkv,
                use_cache=True,
                return_dict=True,
            )

        current_pkv = out.past_key_values
        current_logits = out.logits

    return current_generated, current_attn, current_pkv, current_logits


def main():
    parser = argparse.ArgumentParser(description="Part 2: Save prompt KV and resume generation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default="Explain how saving prompt KV can speed up recovery after a failure.")
    parser.add_argument("--tokens-before-checkpoint", type=int, default=8)
    parser.add_argument("--tokens-after-checkpoint", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default="results/part2_resume")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = choose_device()
    print(f"[INFO] Device: {device}")

    tokenizer = load_tokenizer(args.model)
    model, load_sec = load_model(args.model, device)
    print(f"[INFO] Model loaded in {load_sec:.2f}s")

    prompt = format_prompt(args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = move_batch(inputs, device)

    base_input_ids = inputs["input_ids"]
    base_attention = inputs["attention_mask"]

    # Build prompt state
    with torch.no_grad():
        prompt_out = model(
            input_ids=base_input_ids,
            attention_mask=base_attention,
            use_cache=True,
            return_dict=True,
        )

    prompt_pkv_cpu = cpu_clone_past(prompt_out.past_key_values)
    prompt_logits_cpu = prompt_out.logits.detach().to("cpu").clone()

    checkpoint_path = os.path.join(args.out_dir, "prompt_kv.pt")
    torch.save(
        {
            "past_key_values": prompt_pkv_cpu,
            "prompt_logits": prompt_logits_cpu,
        },
        checkpoint_path,
    )
    print(f"[INFO] Saved prompt KV checkpoint to: {checkpoint_path}")

    total_tokens = args.tokens_before_checkpoint + args.tokens_after_checkpoint

    # Baseline full generation
    t0 = time.time()
    baseline_generated, _, _, _ = greedy_generate_from_state(
        model=model,
        generated=base_input_ids,
        attn=base_attention,
        pkv=prompt_out.past_key_values,
        logits=prompt_out.logits,
        num_steps=total_tokens,
    )
    baseline_time = time.time() - t0

    # Generate only until failure point
    t1 = time.time()
    before_failure_generated, _, _, _ = greedy_generate_from_state(
        model=model,
        generated=base_input_ids,
        attn=base_attention,
        pkv=prompt_out.past_key_values,
        logits=prompt_out.logits,
        num_steps=args.tokens_before_checkpoint,
    )
    before_ckpt_time = time.time() - t1

    # Recover from saved prompt checkpoint and replay suffix
    restored = torch.load(checkpoint_path)
    restored_pkv = move_past_to_device(restored["past_key_values"], device)
    restored_logits = restored["prompt_logits"].to(device)

    replay_time_start = time.time()

    # Replay the tokens before failure from the saved prompt state
    replayed_generated, replayed_attn, replayed_pkv, replayed_logits = greedy_generate_from_state(
        model=model,
        generated=base_input_ids,
        attn=base_attention,
        pkv=restored_pkv,
        logits=restored_logits,
        num_steps=args.tokens_before_checkpoint,
    )

    # Continue generation after recovery
    recovered_generated, _, _, _ = greedy_generate_from_state(
        model=model,
        generated=replayed_generated,
        attn=replayed_attn,
        pkv=replayed_pkv,
        logits=replayed_logits,
        num_steps=args.tokens_after_checkpoint,
    )

    recovery_time = time.time() - replay_time_start

    baseline_text = tokenizer.decode(baseline_generated[0], skip_special_tokens=True)
    recovered_text = tokenizer.decode(recovered_generated[0], skip_special_tokens=True)
    exact_match = torch.equal(baseline_generated, recovered_generated)

    print("\n[RESULT]")
    print(f"Exact token match after recovery: {exact_match}")
    print(f"Baseline full generation time: {baseline_time:.4f}s")
    print(f"Generate-before-failure time: {before_ckpt_time:.4f}s")
    print(f"Recovery path time: {recovery_time:.4f}s")

    save_json(
        {
            "model": args.model,
            "device": device,
            "load_time_sec": round(load_sec, 3),
            "prompt": args.prompt,
            "tokens_before_checkpoint": args.tokens_before_checkpoint,
            "tokens_after_checkpoint": args.tokens_after_checkpoint,
            "baseline_time_sec": round(baseline_time, 4),
            "before_failure_time_sec": round(before_ckpt_time, 4),
            "recovery_time_sec": round(recovery_time, 4),
            "exact_token_match": bool(exact_match),
            "checkpoint_path": checkpoint_path,
            "baseline_text": baseline_text,
            "recovered_text": recovered_text,
        },
        os.path.join(args.out_dir, "part2_resume_summary.json"),
    )
    print(f"[INFO] Saved summary to: {os.path.join(args.out_dir, 'part2_resume_summary.json')}")


if __name__ == "__main__":
    main()
