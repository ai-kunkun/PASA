#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python generation.py \
  --watermark_model "meta-llama/Llama-2-13b-hf" \
  --auxilary_model "meta-llama/Llama-2-7b-hf" \
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
