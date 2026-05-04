
from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _infer_dtype(dtype: Union[str, torch.dtype] = "auto") -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    name = str(dtype).lower()
    if name == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(
    model_name_or_path: str,
    *,
    device_map: Union[str, Dict[str, Any]] = "auto",
    dtype: Union[str, torch.dtype] = "auto",
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
):
    """Load a causal language model and tokenizer for generation/detection."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_infer_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _iter_train_split(dataset):
    if isinstance(dataset, dict) and "train" in dataset:
        return dataset["train"]
    return dataset


def _first_sentences(text: str, n: int = 2) -> str:
    try:
        import nltk

        sentences = nltk.sent_tokenize(text)
        if sentences:
            return " ".join(sentences[:n])
    except Exception:
        pass

    return text[:200]


def pre_process(dataset, min_length: int = 200, data_size: int = 500) -> List[Dict[str, str]]:
    """Collect long samples from a C4-style dataset.

    Returns:
        text: prompt from the first sentences.
        text0: original full text.
    """
    samples: List[Dict[str, str]] = []
    min_length = int(min_length)
    data_size = int(data_size)

    for item in _iter_train_split(dataset):
        if len(samples) >= data_size:
            break

        text = item.get("text", "") if isinstance(item, dict) else ""
        if not isinstance(text, str) or len(text.split()) < min_length:
            continue

        samples.append({
            "text": _first_sentences(text, n=2),
            "text0": text,
        })

    return samples
