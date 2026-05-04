# PASA: A Principled Embedding-Space Watermarking Approach for LLM-Generated Text under Semantic-Invariant Attacks

<p align="center">
  <b>ICML 2026</b>
</p>

<p align="center">
  <b>Zhenxin Ai, Haiyun He</b><br>
  The Hong Kong University of Science and Technology (Guangzhou)
</p>

<p align="center">
  <a href="https://your-paper-link.com">
    <img src="https://img.shields.io/badge/Paper-PDF-red.svg">
  </a>
  <a href="https://your-project-page.com">
    <img src="https://img.shields.io/badge/Project-Page-blue.svg">
  </a>
  <a href="https://github.com/your-name/PASA">
    <img src="https://img.shields.io/badge/Code-GitHub-black.svg">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Conference-ICML%202026-green.svg">
  </a>
</p>

<p align="center">
  <img src="assets/overview.png" width="850">
</p>

## Overview

This repository provides the official implementation of **PASA**, a principled embedding-space watermarking approach for LLM-generated text under semantic-invariant attacks.

PASA embeds watermarks at the semantic level rather than directly manipulating raw token IDs. It first maps tokens into semantic clusters, aggregates the next-token distribution into a cluster-level distribution, and then performs key-guided auxiliary sampling to generate watermarked text. During detection, PASA reconstructs the auxiliary sampling process using a shared secret key and an auxiliary language model.

PASA is designed to improve robustness against semantic-invariant attacks such as paraphrasing and synonym substitution, while maintaining a clean and lightweight generation and detection pipeline.

---

## News

- **[2026.05]** PASA has been accepted to **ICML 2026**.
- Code and project page will be continuously updated.

---

## Navigation

- [Paper](https://your-paper-link.com)
- [Project Page](https://your-project-page.com)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Token-to-Cluster Mapping](#token-to-cluster-mapping)
- [Python API](#python-api)
- [Output Format](#output-format)
- [Citation](#citation)

> Please replace `https://your-paper-link.com` and `https://your-project-page.com` with the official paper and project-page URLs.

---

## Repository Structure

```text
PASA/
├── experiment.sh      # Example script for running generation and detection
├── generation.py      # Main script for watermark generation and detection
├── utils.py           # Utility functions for loading models and preprocessing data
├── watermark.py       # Core PASA watermark implementation
└── README.md
```

---

## Installation

Clone this repository:

```bash
git clone https://github.com/your-name/PASA.git
cd PASA
```

Create a conda environment:

```bash
conda create -n pasa python=3.10 -y
conda activate pasa
```

Install dependencies:

```bash
pip install torch transformers accelerate datasets pandas tqdm nltk tokenizers
```

If you use NLTK sentence tokenization, please run:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Quick Start

We provide an example script `experiment.sh`.

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python generation.py \
  --watermark_model "/path/to/watermark_model" \
  --auxilary_model "/path/to/auxiliary_model" \
  --c4_dataset_path "/path/to/c4_dataset" \
  --c4_subset "realnewslike" \
  --key_map_path "/path/to/token_to_cluster_mapping.csv" \
  --output_csv "./outputs/results.csv" \
  --num_samples 200 \
  --num_semantic_keys 4 \
  --alpha 0.4 \
  --key 123 \
  --top_p 1.0 \
  --start 3 \
  --window_size 3 \
  --T 1.0 \
  --max_new_tokens 300 \
  --min_new_tokens 200
```

Run:

```bash
chmod +x experiment.sh
./experiment.sh
```

---

## Path Configuration

Before running the script, please replace the following paths:

| Placeholder | Description |
|---|---|
| `/path/to/watermark_model` | Path or Hugging Face model name of the language model used for watermarked generation. |
| `/path/to/auxiliary_model` | Path or Hugging Face model name of the auxiliary model used for detection. |
| `/path/to/c4_dataset` | Local path to the C4 dataset directory. |
| `/path/to/token_to_cluster_mapping.csv` | Path to the token-to-cluster mapping CSV file. |
| `./outputs/results.csv` | Path to save the generated results. |

Example:

```bash
python generation.py \
  --watermark_model "/models/Llama-2-13b-chat-hf" \
  --auxilary_model "/models/TinyLlama-1.1B-Chat-v1.0" \
  --c4_dataset_path "/data/c4" \
  --c4_subset "realnewslike" \
  --key_map_path "./data/token_to_cluster_mapping.csv" \
  --output_csv "./outputs/results.csv" \
  --num_samples 200 \
  --num_semantic_keys 4 \
  --alpha 0.4 \
  --key 123 \
  --top_p 1.0 \
  --start 3 \
  --window_size 3 \
  --T 1.0 \
  --max_new_tokens 300 \
  --min_new_tokens 200
```

> Note: the argument name `--auxilary_model` is kept for compatibility with the released code.

---

## Token-to-Cluster Mapping

PASA requires a token-to-cluster mapping file in CSV format.

The CSV file should contain at least the following two columns:

```text
token_id,best_cluster_idx
```

Example:

```csv
token_id,best_cluster_idx
0,1
1,3
2,0
3,2
```

Here:

- `token_id` is the tokenizer vocabulary index.
- `best_cluster_idx` is the assigned semantic cluster ID.
- `best_cluster_idx` should be in `[0, num_semantic_keys - 1]`.

For example, if your mapping file contains cluster IDs from `0` to `3`, then use:

```bash
--num_semantic_keys 4
```

---

## Main Arguments

| Argument | Description |
|---|---|
| `--watermark_model` | Language model used for watermarked and unwatermarked generation. |
| `--auxilary_model` | Auxiliary language model used for watermark detection. |
| `--c4_dataset_path` | Local path to the C4 dataset directory. |
| `--c4_subset` | C4 subset name, e.g., `realnewslike`. |
| `--key_map_path` | Path to the token-to-cluster mapping CSV file. |
| `--output_csv` | Path to the output CSV file. |
| `--num_samples` | Number of samples to generate and evaluate. |
| `--num_semantic_keys` | Number of semantic clusters. |
| `--alpha` | Truncation threshold for the auxiliary distribution. |
| `--key` | Secret watermark key. The same key should be used for generation and detection. |
| `--top_p` | Nucleus sampling parameter. |
| `--start` | Token index from which watermarking starts. |
| `--window_size` | Context window size used to build the semantic seed. |
| `--T` | Sampling temperature. |
| `--max_new_tokens` | Maximum number of generated tokens. |
| `--min_new_tokens` | Minimum number of generated tokens before EOS is allowed. |

---

## Python API

You can also use PASA directly in Python.

```python
import torch

from utils import load_model
from watermark import Watermark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

watermark_model, watermark_tokenizer = load_model("/path/to/watermark_model")
auxilary_model, auxilary_tokenizer = load_model("/path/to/auxiliary_model")

watermark = Watermark(
    device=device,
    watermark_tokenizer=watermark_tokenizer,
    watermark_model=watermark_model,
    auxilary_tokenizer=auxilary_tokenizer,
    auxilary_model=auxilary_model,
    key_map_path="/path/to/token_to_cluster_mapping.csv",
    num_semantic_keys=4,
    alpha=0.4,
    top_p=1.0,
    key=123,
    T=1.0,
    start=3,
    window_size=3,
    max_new_tokens=300,
    min_new_tokens=200,
)

prompt = "Large language models have shown remarkable capabilities in"

uwm_text = watermark.generate_unwatermarked(prompt)
wm_text = watermark.generate_watermarked(prompt)

uwm_score = watermark.detection(uwm_text)
wm_score = watermark.detection(wm_text)

print("Unwatermarked text:")
print(uwm_text)

print("\nWatermarked text:")
print(wm_text)

print("\nUnwatermarked score:", uwm_score)
print("Watermarked score:", wm_score)
```

---

## Output Format

The generation script saves the results into a CSV file.

Typical columns include:

| Column | Description |
|---|---|
| `text_id` | Sample index. |
| `prompt` | Prompt used for generation. |
| `human` | Human-written reference text. |
| `uwm_text` | Unwatermarked generated text. |
| `wm_text` | Watermarked generated text. |
| `human_hit_rate` | Detection score of human text. |
| `uwm_hit_rate` | Detection score of unwatermarked generated text. |
| `wm_hit_rate` | Detection score of watermarked generated text. |
| `human_hits` / `human_count` | Raw hit statistics for human text. |
| `uwm_hits` / `uwm_count` | Raw hit statistics for unwatermarked generated text. |
| `wm_hits` / `wm_count` | Raw hit statistics for watermarked generated text. |
| `error` | Error message if a sample fails. |

---

## Method

PASA consists of three main components.

### 1. Semantic Clustering

PASA first constructs a semantic mapping from token IDs to semantic clusters:

```text
f: token_id -> semantic_cluster_id
```

This mapping allows the watermark to operate in the semantic space instead of the raw token space.

### 2. Watermark Generation

At each generation step, PASA obtains the next-token distribution from the language model and aggregates it into a cluster-level distribution.

Then, PASA constructs an auxiliary distribution with a truncation threshold `alpha` and an overflow state. A secret-key-based pseudo-random function samples an auxiliary semantic key. If the sampled key corresponds to a semantic cluster, the next token is sampled from that cluster. If it corresponds to the overflow state, the next token is sampled from the residual distribution.

### 3. Watermark Detection

During detection, PASA uses an auxiliary language model to reconstruct the cluster-level distribution and replay the auxiliary sampling process with the shared secret key.

The detection score is computed by comparing the replayed semantic key with the semantic cluster of each observed token.

---

## Tips

### Use compatible tokenizers

The token-to-cluster mapping is tokenizer-dependent. For better performance, the watermark generation model and the auxiliary detection model should use compatible tokenizers.

### Check the mapping file

If the detection score is unexpectedly low, first check the mapping file:

```bash
head /path/to/token_to_cluster_mapping.csv
```

The file should contain:

```text
token_id,best_cluster_idx
```

Also make sure that `best_cluster_idx` is in the range:

```text
[0, num_semantic_keys - 1]
```

### Out-of-memory issues

If the model is too large for a single GPU, try:

```bash
--max_new_tokens 200
--min_new_tokens 100
```

You can also use a smaller auxiliary model or enable automatic model dispatching with Hugging Face `device_map="auto"`.

---

## Common Issues

### `FileNotFoundError: mapping file does not exist`

Please check whether `--key_map_path` points to a valid CSV file.

### `CUDA out of memory`

Try using a smaller model or fewer generated tokens:

```bash
--max_new_tokens 200
--min_new_tokens 100
```

### Very low watermark score

Please check:

- whether the correct secret key is used;
- whether the mapping file matches the tokenizer;
- whether `--num_semantic_keys` matches the mapping file;
- whether the generation model and auxiliary model use compatible tokenizers.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{ai2026pasa,
  title     = {PASA: A Principled Embedding-Space Watermarking Approach for LLM-Generated Text under Semantic-Invariant Attacks},
  author    = {Ai, Zhenxin and He, Haiyun},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

---

## Acknowledgement

This repository is inspired by recent research on robust watermarking for large language models.

---

## License

This project is released under the MIT License.
