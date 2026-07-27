#!/usr/bin/env bash
# Quick benchmark — 1MB corpus, few iterations, all 4 implementations.
#
# Usage:  bash benchmarks/quick_bench.sh
#
# Output: final stdout is a complete markdown report (hardware + results).
#   bash benchmarks/quick_bench.sh > results.md  to capture.
# Progress/debug messages go to stderr.

set -euo pipefail
cd "$(dirname "$0")/.."
BMDIR="benchmarks"
RESULTS_DIR="$BMDIR/results"
CORPUS="corpus_1MB.txt"
CORPUS_PATH="$BMDIR/$CORPUS"

# Set up dependencies (stdout → stderr)
source "$BMDIR/setup_bench_env.sh" >&2

# Build _mbpe.so if stale
pixi run mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/_mbpe.so 2>/dev/null

# Generate corpora if missing
if [ ! -f "$CORPUS_PATH" ]; then
    "$VENV_PYTHON" "$BMDIR/generate_corpora.py" >&2
fi

mkdir -p "$RESULTS_DIR"
export BPE_CORPUS="$CORPUS_PATH"

echo "═══════════════════════════════════════════" >&2
echo " Quick Benchmark — 1MB corpus" >&2
echo "═══════════════════════════════════════════" >&2

MOJO_OUT="$RESULTS_DIR/mojo_1MB.json"
PY_OUT="$RESULTS_DIR/py_1MB.json"
MBPE_OUT="$RESULTS_DIR/mbpe_1MB.json"
RS_OUT="$RESULTS_DIR/rs_1MB.json"
NATIVE_OUT="$RESULTS_DIR/native_1MB.json"

echo "  [1/6] Mojo (training pipeline)..." >&2
pixi run mojo -I . "$BMDIR/bm.mojo" > "$MOJO_OUT" 2>/dev/null

echo "  [2/6] Mojo native (pre-built vocabs)..." >&2
pixi run mojo -I . "$BMDIR/bm_pretrained.mojo" > "$NATIVE_OUT" 2>/dev/null

echo "  [3/6] Python tiktoken..." >&2
"$VENV_PYTHON" "$BMDIR/benchmark_tiktoken.py" > "$PY_OUT" 2>/dev/null

echo "  [4/6] mbpe Python bindings (quick, n_iters=3)..." >&2
pixi run python "$BMDIR/benchmark_mbpe_quick.py" > "$MBPE_OUT" 2>/dev/null

echo "  [5/6] Rust tiktoken-rs..." >&2
(cd "$BMDIR/benchmark_rust" && cargo build --release 2>&1 | tail -1 >&2)
"$BMDIR/benchmark_rust/target/release/benchmark_rust" > "$RS_OUT" 2>/dev/null

echo "  [6/6] Generating markdown report..." >&2
echo "" >&2

# ── Markdown report (-> stdout) ─────────────────────────────────
"$VENV_PYTHON" "$BMDIR/collate.py" "$MOJO_OUT" "$PY_OUT" "$RS_OUT" "$MBPE_OUT" "$NATIVE_OUT"
