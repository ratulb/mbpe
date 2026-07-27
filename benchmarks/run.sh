#!/usr/bin/env bash
# Full benchmark suite — 4 implementations × 6 corpus sizes × 4 vocab sizes.
#
# Usage:  bash benchmarks/run.sh
#   BPE_NO_RUST=1  skip Rust benchmark
#   BPE_SKIP_PY=1  skip Python benchmark
#
# Output: final stdout is a complete markdown report (hardware + results).
#   bash benchmarks/run.sh > results.md  to capture.
# Progress/debug messages go to stderr.

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

echo "╔══════════════════════════════════════════════════════════════╗" >&2
echo "║  simple_bpe  —  Multi-Language Benchmark Suite              ║" >&2
echo "╚══════════════════════════════════════════════════════════════╝" >&2

# ── Setup dependencies (stdout → stderr) ────────────────────────
source "$BMDIR/setup_bench_env.sh" >&2

echo "" >&2
echo "── Mojo ──" >&2
if command -v mojo &>/dev/null; then
    echo "  Mojo: $(mojo --version 2>&1 | head -1 | cut -d' ' -f3)" >&2
else
    echo "  ERROR: Mojo not found. Activate pixi shell first." >&2
    exit 1
fi

echo "  Building _mbpe.so..." >&2
pixi run mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/_mbpe.so 2>&1 | tail -1 >&2
echo "  _mbpe.so built" >&2

# ── Generate corpora (stdout → stderr) ──────────────────────────
echo "" >&2
echo "── Corpora ──" >&2
"$VENV_PYTHON" "$BMDIR/generate_corpora.py" >&2

# ── Run benchmarks ───────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

for corpus_spec in "${CORPORA[@]}"; do
    CORPUS_FILE="${corpus_spec%%:*}"
    CORPUS_LABEL="${corpus_spec##*:}"
    CORPUS_PATH="$BMDIR/$CORPUS_FILE"

    echo "" >&2
    echo "═══════════════════════════════════════════════════════════" >&2
    echo "  Corpus: $CORPUS_LABEL ($(wc -c < "$CORPUS_PATH") bytes)" >&2
    echo "═══════════════════════════════════════════════════════════" >&2

    export BPE_CORPUS="$CORPUS_PATH"

    MOJO_OUT="$RESULTS_DIR/mojo_${CORPUS_LABEL}.json"
    PY_OUT="$RESULTS_DIR/py_${CORPUS_LABEL}.json"
    MBPE_OUT="$RESULTS_DIR/mbpe_${CORPUS_LABEL}.json"
    RS_OUT="$RESULTS_DIR/rs_${CORPUS_LABEL}.json"

    # Mojo
    echo "  [1/4] Mojo..." >&2
    pixi run mojo -I . "$BMDIR/bm.mojo" > "$MOJO_OUT" 2>/dev/null
    echo "    $(wc -l < "$MOJO_OUT") results" >&2

    # Python tiktoken
    echo "  [2/4] Python tiktoken..." >&2
    "$VENV_PYTHON" "$BMDIR/benchmark_tiktoken.py" > "$PY_OUT" 2>/dev/null
    echo "    $(wc -l < "$PY_OUT") results" >&2

    # mbpe Python bindings
    echo "  [3/4] mbpe Python bindings..." >&2
    pixi run python "$BMDIR/benchmark_mbpe.py" > "$MBPE_OUT" 2>/dev/null
    echo "    $(wc -l < "$MBPE_OUT") results" >&2

    # Rust tiktoken-rs
    if [ -z "${BPE_NO_RUST:-}" ]; then
        echo "  [4/4] Rust tiktoken-rs..." >&2
        (cd "$BMDIR/benchmark_rust" && cargo build --release 2>&1 | tail -1 >&2)
        "$BMDIR/benchmark_rust/target/release/benchmark_rust" > "$RS_OUT" 2>/dev/null
        echo "    $(wc -l < "$RS_OUT") results" >&2
    else
        echo "  [4/4] Rust tiktoken-rs: skipped (BPE_NO_RUST=1)" >&2
        printf '' > "$RS_OUT"
    fi
done

# ── Markdown report (-> stdout) ─────────────────────────────────
echo "" >&2
echo "═══════════════════════════════════════════════════════════" >&2
echo "  Generating markdown report..." >&2
echo "═══════════════════════════════════════════════════════════" >&2

# Hardware info once at the top
"$VENV_PYTHON" "$BMDIR/hardware_info.py"

for corpus_spec in "${CORPORA[@]}"; do
    CORPUS_LABEL="${corpus_spec##*:}"
    MOJO_OUT="$RESULTS_DIR/mojo_${CORPUS_LABEL}.json"
    PY_OUT="$RESULTS_DIR/py_${CORPUS_LABEL}.json"
    MBPE_OUT="$RESULTS_DIR/mbpe_${CORPUS_LABEL}.json"
    RS_OUT="$RESULTS_DIR/rs_${CORPUS_LABEL}.json"

    echo ""
    "$VENV_PYTHON" "$BMDIR/collate.py" --no-hardware "$MOJO_OUT" "$PY_OUT" "$RS_OUT" "$MBPE_OUT"
done

echo "" >&2
echo "Done. Raw results in $RESULTS_DIR/" >&2
echo "" >&2
