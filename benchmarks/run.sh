#!/usr/bin/env bash
# Runner for simple_bpe benchmarks.
# Orchestrates Mojo, Python tiktoken, and Rust tiktoken-rs across
# multiple corpus sizes and vocab sizes.
#
# Usage:  bash benchmarks/run.sh
#   BPE_NO_RUST=1  skip Rust benchmark
#   BPE_SKIP_PY=1  skip Python benchmark

set -euo pipefail
cd "$(dirname "$0")/.."
BMDIR="benchmarks"

# ── Config ───────────────────────────────────────────────────────
CORPORA=(
    "corpus_10KB.txt:10KB"
    "corpus_100KB.txt:100KB"
    "corpus_500KB.txt:500KB"
    "corpus_1MB.txt:1MB"
    "corpus_2MB.txt:2MB"
    "corpus_5MB.txt:5MB"
)
RESULTS_DIR="$BMDIR/results"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  simple_bpe  —  Multi-Language Benchmark Suite              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Setup dependencies ──────────────────────────────────────────
source "$BMDIR/setup_bench_env.sh"

echo ""
echo "── Mojo ──"
if command -v mojo &>/dev/null; then
    echo "  Mojo: $(mojo --version 2>&1 | head -1 | cut -d' ' -f3)"
else
    echo "  ERROR: Mojo not found. Activate pixi shell first."
    exit 1
fi

# Build mbpe Python bindings (shared library)
echo "  Building _mbpe.so..."
pixi run mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/_mbpe.so 2>&1 | tail -1
echo "  _mbpe.so built"

# ── Generate corpora ─────────────────────────────────────────────
echo ""
echo "── Corpora ──"
"$VENV_PYTHON" "$BMDIR/generate_corpora.py" 2>&1

# ── Run benchmarks ───────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

for corpus_spec in "${CORPORA[@]}"; do
    CORPUS_FILE="${corpus_spec%%:*}"
    CORPUS_LABEL="${corpus_spec##*:}"
    CORPUS_PATH="$BMDIR/$CORPUS_FILE"

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Corpus: $CORPUS_LABEL ($(wc -c < "$CORPUS_PATH") bytes)"
    echo "═══════════════════════════════════════════════════════════"

    export BPE_CORPUS="$CORPUS_PATH"

    MOJO_OUT="$RESULTS_DIR/mojo_${CORPUS_LABEL}.json"
    PY_OUT="$RESULTS_DIR/py_${CORPUS_LABEL}.json"
    MBPE_OUT="$RESULTS_DIR/mbpe_${CORPUS_LABEL}.json"
    RS_OUT="$RESULTS_DIR/rs_${CORPUS_LABEL}.json"

    # Mojo
    echo "  [1/4] Mojo..."
    pixi run mojo -I . "$BMDIR/bm.mojo" > "$MOJO_OUT" 2>/dev/null
    echo "    $(wc -l < "$MOJO_OUT") results"

    # Python tiktoken
    echo "  [2/4] Python tiktoken..."
    "$VENV_PYTHON" "$BMDIR/benchmark_tiktoken.py" > "$PY_OUT" 2>/dev/null
    echo "    $(wc -l < "$PY_OUT") results"

    # mbpe Python bindings
    echo "  [3/4] mbpe Python bindings..."
    pixi run python "$BMDIR/benchmark_mbpe.py" > "$MBPE_OUT" 2>/dev/null
    echo "    $(wc -l < "$MBPE_OUT") results"

    # Rust tiktoken-rs
    if [ -z "${BPE_NO_RUST:-}" ]; then
        echo "  [4/4] Rust tiktoken-rs..."
        (cd "$BMDIR/benchmark_rust" && cargo build --release 2>&1 | tail -1)
        "$BMDIR/benchmark_rust/target/release/benchmark_rust" > "$RS_OUT" 2>/dev/null
        echo "    $(wc -l < "$RS_OUT") results"
    else
        echo "  [4/4] Rust tiktoken-rs: skipped (BPE_NO_RUST=1)"
        printf '' > "$RS_OUT"
    fi
done

# ── Collate ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results"
echo "═══════════════════════════════════════════════════════════"

for corpus_spec in "${CORPORA[@]}"; do
    CORPUS_LABEL="${corpus_spec##*:}"
    MOJO_OUT="$RESULTS_DIR/mojo_${CORPUS_LABEL}.json"
    PY_OUT="$RESULTS_DIR/py_${CORPUS_LABEL}.json"
    MBPE_OUT="$RESULTS_DIR/mbpe_${CORPUS_LABEL}.json"
    RS_OUT="$RESULTS_DIR/rs_${CORPUS_LABEL}.json"

    echo ""
    "$VENV_PYTHON" "$BMDIR/collate.py" "$MOJO_OUT" "$PY_OUT" "$RS_OUT" "$MBPE_OUT"
done

echo ""
echo "Done. Raw results in $RESULTS_DIR/"
echo ""
