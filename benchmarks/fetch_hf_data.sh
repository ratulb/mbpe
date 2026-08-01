#!/usr/bin/env bash
# Fetch the tokenizer.json files used by the HF tokenizers benchmark
# (benchmarks/benchmark_hf/). Run from the repo root. Gitignored data.
set -euo pipefail
mkdir -p benchmarks/hf_data
cd benchmarks/hf_data
curl -sfL "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json" -o gpt2.json
curl -sfL "https://huggingface.co/BEE-spoke-data/cl100k_base/resolve/main/tokenizer.json" -o cl100k.json
curl -sfL "https://huggingface.co/tokiers/o200k/resolve/main/tokenizer.json" -o o200k.json
echo "Fetched gpt2/cl100k/o200k tokenizer.json into benchmarks/hf_data/"
