#!/usr/bin/env bash
# Quick benchmark — 1MB corpus, few iterations, all 4 implementations.
# Usage:  bash benchmarks/quick_bench.sh

set -euo pipefail
cd "$(dirname "$0")/.."
BMDIR="benchmarks"
RESULTS_DIR="$BMDIR/results"
CORPUS="corpus_1MB.txt"
CORPUS_PATH="$BMDIR/$CORPUS"

# Set up dependencies (Rust + tiktoken venv in /tmp)
source "$BMDIR/setup_bench_env.sh"

# Build mbpe.so if stale
pixi run mojo build mbpe.mojo --emit shared-lib -o mbpe.so 2>/dev/null

# Generate corpora if missing
if [ ! -f "$CORPUS_PATH" ]; then
    "$VENV_PYTHON" "$BMDIR/generate_corpora.py" 2>/dev/null
fi

mkdir -p "$RESULTS_DIR"
export BPE_CORPUS="$CORPUS_PATH"

echo "═══════════════════════════════════════════"
echo " Quick Benchmark — 1MB corpus"
echo "═══════════════════════════════════════════"

MOJO_OUT="$RESULTS_DIR/mojo_1MB.json"
PY_OUT="$RESULTS_DIR/py_1MB.json"
MBPE_OUT="$RESULTS_DIR/mbpe_1MB.json"
RS_OUT="$RESULTS_DIR/rs_1MB.json"

echo "  [1/5] Mojo..."
pixi run mojo -I . "$BMDIR/bm.mojo" > "$MOJO_OUT" 2>/dev/null

echo "  [2/5] Python tiktoken..."
"$VENV_PYTHON" "$BMDIR/benchmark_tiktoken.py" > "$PY_OUT" 2>/dev/null

echo "  [3/5] mbpe Python bindings (quick, n_iters=3)..."
pixi run python "$BMDIR/benchmark_mbpe_quick.py" > "$MBPE_OUT" 2>/dev/null

echo "  [4/5] Rust tiktoken-rs..."
( cd "$BMDIR/benchmark_rust" && cargo build --release 2>&1 | tail -1 )
"$BMDIR/benchmark_rust/target/release/benchmark_rust" > "$RS_OUT" 2>/dev/null

echo "  [5/5] Collating..."
echo ""
"$VENV_PYTHON" "$BMDIR/collate.py" "$MOJO_OUT" "$PY_OUT" "$RS_OUT" "$MBPE_OUT"
