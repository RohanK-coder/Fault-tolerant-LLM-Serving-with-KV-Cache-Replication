import json
import os
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: str):
    dtype = torch.float16 if device == "mps" else torch.float32
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    load_sec = time.time() - start
    return model, load_sec


def format_prompt(user_prompt: str) -> str:
    if "[INST]" in user_prompt:
        return user_prompt
    return f"<|system|>You are a helpful assistant.</s><|user|>{user_prompt}</s><|assistant|>"


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def summarize_past_key_values(past_key_values) -> Tuple[List[Dict], int]:
    layers = []
    total_kv_bytes = 0
    for layer_idx, layer_kv in enumerate(past_key_values):
        key = layer_kv[0].detach().to("cpu")
        value = layer_kv[1].detach().to("cpu")
        key_bytes = tensor_nbytes(key)
        value_bytes = tensor_nbytes(value)
        total_kv_bytes += key_bytes + value_bytes
        layers.append(
            {
                "layer": layer_idx,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
                "key_bytes": key_bytes,
                "value_bytes": value_bytes,
            }
        )
    return layers, total_kv_bytes


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def decode_new_tokens(tokenizer, full_sequence: torch.Tensor, prompt_len: int) -> str:
    new_ids = full_sequence[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
