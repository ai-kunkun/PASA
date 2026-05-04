#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import os
from typing import Iterable, List

import nltk
import torch
from datasets import load_dataset
from tqdm import tqdm

from utils import load_model
from watermark import Watermark


def sorted_glob(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def resolve_local_c4_train_files(base_dir: str, subset: str = "realnewslike") -> List[str]:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"C4 directory does not exist: {base_dir}")

    patterns = [
        "c4-train.*.json.gz",
        "c4-train*.json.gz",
        "c4-train.*.json",
        "c4-train*.json",
        "train.*.json.gz",
        "train*.json.gz",
        "train.*.json",
        "train*.json",
    ]

    tried = []
    for directory in [os.path.join(base_dir, subset), base_dir]:
        if not os.path.isdir(directory):
            continue
        for pattern in patterns:
            path_pattern = os.path.join(directory, pattern)
            tried.append(path_pattern)
            files = sorted_glob(path_pattern)
            if files:
                return files

    message = "No local C4 JSON shards found. Tried:\n" + "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(message)


def load_local_c4_streaming(base_dir: str, subset: str = "realnewslike"):
    files = resolve_local_c4_train_files(base_dir, subset)
    return load_dataset("json", data_files={"train": files}, split="train", streaming=True)


def iter_texts(dataset: Iterable[dict], num_samples: int, min_words: int) -> Iterable[str]:
    count = 0
    min_words = max(1, int(min_words))
    for item in dataset:
        text = item.get("text", "") if isinstance(item, dict) else ""
        if isinstance(text, str) and len(text.split()) >= min_words:
            yield text
            count += 1
            if count >= num_samples:
                break


def build_prompt(text: str, fallback_chars: int = 200) -> str:
    try:
        sentences = nltk.sent_tokenize(text)
        if sentences:
            return " ".join(sentences[:2])
    except Exception:
        pass
    return text[:fallback_chars]


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_results(path: str, rows: Iterable[dict], header: list[str]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            f.flush()


def make_watermark(args, device, watermark_model, watermark_tokenizer, auxilary_model, auxilary_tokenizer) -> Watermark:
    return Watermark(
        device=device,
        watermark_tokenizer=watermark_tokenizer,
        watermark_model=watermark_model,
        auxilary_tokenizer=auxilary_tokenizer,
        auxilary_model=auxilary_model,
        key_map_path=args.key_map_path,
        num_semantic_keys=args.num_semantic_keys,
        alpha=args.alpha,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        key=args.key,
        T=args.T,
        start=args.start,
        window_size=args.window_size,
        verbose=args.verbose,
    )


def generate_rows(args, watermark: Watermark, dataset) -> Iterable[dict]:
    texts = iter_texts(dataset, args.num_samples, args.min_new_tokens)
    for text_id, human in enumerate(tqdm(texts, total=args.num_samples, desc="Generating")):
        try:
            prompt = build_prompt(human)
            uwm_text = watermark.generate_unwatermarked(prompt)
            wm_text = watermark.generate_watermarked(prompt)

            human_score, human_hits, human_count = watermark.detection(human)
            uwm_score, uwm_hits, uwm_count = watermark.detection(uwm_text)
            wm_score, wm_hits, wm_count = watermark.detection(wm_text)

            yield {
                "text_id": text_id,
                "prompt": prompt,
                "human": human,
                "uwm_text": uwm_text,
                "wm_text": wm_text,
                "human_hit_rate": f"{human_score:.4f}",
                "human_hits": human_hits,
                "human_count": human_count,
                "uwm_hit_rate": f"{uwm_score:.4f}",
                "uwm_hits": uwm_hits,
                "uwm_count": uwm_count,
                "wm_hit_rate": f"{wm_score:.4f}",
                "wm_hits": wm_hits,
                "wm_count": wm_count,
                "error": "",
            }
        except Exception as exc:
            yield {
                "text_id": text_id,
                "prompt": "",
                "human": human,
                "uwm_text": "",
                "wm_text": "",
                "human_hit_rate": "",
                "human_hits": "",
                "human_count": "",
                "uwm_hit_rate": "",
                "uwm_hits": "",
                "uwm_count": "",
                "wm_hit_rate": "",
                "wm_hits": "",
                "wm_count": "",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Run K-cluster watermark generation and detection on local C4 shards.")

    parser.add_argument("--watermark_model", type=str, required=True)
    parser.add_argument("--auxilary_model", type=str, required=True)
    parser.add_argument("--c4_dataset_path", type=str, required=True)
    parser.add_argument("--c4_subset", type=str, default="realnewslike")
    parser.add_argument("--key_map_path", type=str, required=True)

    parser.add_argument("--num_semantic_keys", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--key", type=int, default=123)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=2)

    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--min_new_tokens", type=int, default=200)

    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_csv", type=str, default="output/DAWA_output/output.csv")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def validate_args(args) -> None:
    if not os.path.exists(args.key_map_path):
        raise FileNotFoundError(f"Mapping file does not exist: {args.key_map_path}")
    if args.num_semantic_keys <= 0:
        raise ValueError("num_semantic_keys must be positive")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")


def main():
    args = parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    auxilary_model, auxilary_tokenizer = load_model(args.auxilary_model)
    watermark_model.eval()
    auxilary_model.eval()

    dataset = load_local_c4_streaming(args.c4_dataset_path, args.c4_subset)
    watermark = make_watermark(args, device, watermark_model, watermark_tokenizer, auxilary_model, auxilary_tokenizer)

    header = [
        "text_id",
        "prompt",
        "human",
        "uwm_text",
        "wm_text",
        "human_hit_rate",
        "human_hits",
        "human_count",
        "uwm_hit_rate",
        "uwm_hits",
        "uwm_count",
        "wm_hit_rate",
        "wm_hits",
        "wm_count",
        "error",
    ]
    write_results(args.output_csv, generate_rows(args, watermark, dataset), header)
    print(f"Saved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
