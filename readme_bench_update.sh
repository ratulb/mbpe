#!/usr/bin/env bash
# Single script: benchmark all implementations on a corpus, update README tables.
# Usage:  bash readme_bench_update.sh [corpus_path]
#
# Default corpus: benchmarks/corpus_5MB.txt
# Results written to benchmarks/results/{native,mbpe,tiktoken,tiktoken-rs,training}.json
set -euo pipefail

cd "$(dirname "$0")"

CORPUS="${1:-benchmarks/corpus_5MB.txt}"
RESULTS="benchmarks/results"
mkdir -p "$RESULTS"

echo "=== Running benchmarks on: $CORPUS ==="

echo "  1/5  Mojo native (pre-trained encode/decode only)..."
BPE_CORPUS="$CORPUS" pixi run mojo -I . benchmarks/bm_pretrained.mojo > "$RESULTS/native.json"

echo "  2/5  mbpe Python bindings..."
BPE_CORPUS="$CORPUS" PYTHONPATH=python-binding /tmp/mbpe-bench-venv/bin/python benchmarks/benchmark_mbpe.py > "$RESULTS/mbpe.json"

echo "  3/5  tiktoken (Python)..."
BPE_CORPUS="$CORPUS" /tmp/mbpe-bench-venv/bin/python benchmarks/benchmark_tiktoken.py > "$RESULTS/tiktoken.json"

echo "  4/5  tiktoken-rs..."
BPE_CORPUS="$CORPUS" cargo run --release --manifest-path benchmarks/benchmark_rust/Cargo.toml 2>/dev/null > "$RESULTS/tiktoken-rs.json"

echo "  5/5  Mojo training (GPT4, 4 vocab sizes)..."
BPE_CORPUS="$CORPUS" pixi run mojo -I . benchmarks/bm_train.mojo > "$RESULTS/training.json"

echo ""
echo "=== Updating README.md ==="
bash scripts/update_readme_benchmarks.sh

echo ""
echo "=== Done ==="
