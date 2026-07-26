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

# ── Check dependencies ───────────────────────────────────────────
echo ""
echo "── Dependencies ──"

# Mojo
if command -v mojo &>/dev/null; then
    echo "  Mojo: $(mojo --version 2>&1 | head -1 | cut -d' ' -f3)"
else
    echo "  ERROR: Mojo not found. Activate pixi shell first."
    exit 1
fi

# Python tiktoken
if pixi run python -c "import tiktoken" 2>/dev/null; then
    echo "  Python tiktoken: $(pixi run python -c 'import tiktoken; print(tiktoken.__version__)')"
else
    echo "  Installing tiktoken..."
    pixi run python -m ensurepip --upgrade 2>&1 | tail -1
    pixi run python -m pip install tiktoken 2>&1 | tail -1
fi

# Rust
if command -v cargo &>/dev/null && [ -z "${BPE_NO_RUST:-}" ]; then
    export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$(pixi run which gcc)"
    echo "  Rust: $(cargo --version)  linker: $CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER"
    (cd "$BMDIR/benchmark_rust" && cargo build --release 2>&1 | tail -1)
    echo "  Rust benchmark built"
else
    echo "  Rust: skipped (cargo not found or BPE_NO_RUST=1)"
fi

# ── Generate corpora ─────────────────────────────────────────────
echo ""
echo "── Corpora ──"
pixi run python "$BMDIR/generate_corpora.py" 2>&1

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
    RS_OUT="$RESULTS_DIR/rs_${CORPUS_LABEL}.json"

    # Mojo
    echo "  [1/3] Mojo..."
    pixi run mojo -I . "$BMDIR/bm.mojo" > "$MOJO_OUT" 2>/dev/null
    echo "    $(wc -l < "$MOJO_OUT") results"

    # Python tiktoken
    echo "  [2/3] Python tiktoken..."
    pixi run python "$BMDIR/benchmark_tiktoken.py" > "$PY_OUT" 2>/dev/null
    echo "    $(wc -l < "$PY_OUT") results"

    # Rust tiktoken-rs
    if command -v cargo &>/dev/null && [ -z "${BPE_NO_RUST:-}" ]; then
        echo "  [3/3] Rust tiktoken-rs..."
        source "$HOME/.cargo/env"
        "$BMDIR/benchmark_rust/target/release/benchmark_rust" > "$RS_OUT" 2>/dev/null
        echo "    $(wc -l < "$RS_OUT") results"
    else
        echo "  [3/3] Rust tiktoken-rs: skipped"
        echo "{}" > "$RS_OUT"
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
    RS_OUT="$RESULTS_DIR/rs_${CORPUS_LABEL}.json"

    echo ""
    pixi run python "$BMDIR/collate.py" "$MOJO_OUT" "$PY_OUT" "$RS_OUT"
done

echo ""
echo "Done. Raw results in $RESULTS_DIR/"
echo ""
