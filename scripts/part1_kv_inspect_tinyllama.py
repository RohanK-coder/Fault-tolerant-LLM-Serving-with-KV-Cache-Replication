#!/usr/bin/env python3
import argparse

import torch

from common import (
    DEFAULT_MODEL,
    choose_device,
    decode_new_tokens,
    format_prompt,
    load_model,
    load_tokenizer,
    move_batch,
    save_json,
    summarize_past_key_values,
)


def main():
    parser = argparse.ArgumentParser(description="Part 1: Inspect KV cache on TinyLlama")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default="Explain why KV cache matters in LLM inference.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", type=str, default="results/part1_tinyllama.json")
    args = parser.parse_args()

    device = choose_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.model}")

    tokenizer = load_tokenizer(args.model)
    model, load_sec = load_model(args.model, device)
    print(f"[INFO] Model loaded in {load_sec:.2f}s")

    prompt = format_prompt(args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = move_batch(inputs, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    print(f"[INFO] Prompt tokens: {prompt_len}")

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_full = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
    generated_new = decode_new_tokens(tokenizer, gen.sequences, prompt_len)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)

    layers, total_kv_bytes = summarize_past_key_values(outputs.past_key_values)

    print("\n[GENERATED NEW TEXT]")
    print(generated_new)
    print("\n[KV CACHE INFO]")
    print(f"Number of layers: {len(layers)}")
    for layer in layers:
        print(
            f"Layer {layer['layer']:02d} | "
            f"K {layer['key_shape']} {layer['key_dtype']} | "
            f"V {layer['value_shape']} {layer['value_dtype']}"
        )
    print(f"[INFO] Total KV size: {total_kv_bytes / (1024 * 1024):.4f} MB")

    save_json(
        {
            "model": args.model,
            "device": device,
            "load_time_sec": round(load_sec, 3),
            "prompt": args.prompt,
            "prompt_tokens": prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "generated_new_text": generated_new,
            "generated_full_text": generated_full,
            "num_layers": len(layers),
            "total_kv_bytes": total_kv_bytes,
            "total_kv_mb": round(total_kv_bytes / (1024 * 1024), 4),
            "layers": layers,
        },
        args.out,
    )
    print(f"[INFO] Saved summary to: {args.out}")


if __name__ == "__main__":
    main()
