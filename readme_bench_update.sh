#!/usr/bin/env bash
# Single script: benchmark all implementations on a corpus, update README tables.
# Usage:  bash readme_bench_update.sh [corpus_path]
#
# Default corpus: benchmarks/corpus_5MB.txt
# Results written to benchmarks/results/{native,mbpe,tiktoken,tiktoken-rs,training}.json
#
# Self-sufficient: regenerates gitignored corpora when the requested one is
# missing, and installs a benchmark-only Rust toolchain under /tmp (or
# $MBPE_RUST_HOME) when cargo is not on PATH. Reuses an existing install.
set -euo pipefail

cd "$(dirname "$0")"

CORPUS="${1:-benchmarks/corpus_5MB.txt}"
RESULTS="benchmarks/results"
mkdir -p "$RESULTS"

# ── Ensure the corpus exists ────────────────────────────────────
# Generated corpora (corpus_*KB.txt / corpus_*MB.txt) are gitignored;
# recreate them via benchmarks/generate_corpora.py.
if [ ! -f "$CORPUS" ]; then
    case "$(basename "$CORPUS")" in
        corpus_*KB.txt|corpus_*MB.txt)
            echo "  Corpus missing ($CORPUS) — generating..."
            pixi run -e dev python benchmarks/generate_corpora.py
            ;;
        *)
            echo "  Error: corpus not found: $CORPUS" >&2
            exit 1
            ;;
    esac
fi

# ── Ensure a Rust toolchain exists ──────────────────────────────
# tiktoken-rs is only a benchmark opponent, so the toolchain lives outside
# the repo. NB: export RUSTUP_HOME/CARGO_HOME — prefix-assigned vars on the
# curl pipeline do NOT reach the piped `sh`, and the install would go to ~.
if ! command -v cargo >/dev/null 2>&1; then
    MBPE_RUST_HOME="${MBPE_RUST_HOME:-/tmp/mbpe-rust}"
    export RUSTUP_HOME="$MBPE_RUST_HOME/rustup"
    export CARGO_HOME="$MBPE_RUST_HOME/cargo"
    export PATH="$CARGO_HOME/bin:$PATH"
    if [ -x "$CARGO_HOME/bin/cargo" ]; then
        echo "  Reusing Rust toolchain at $MBPE_RUST_HOME"
    else
        echo "  Installing Rust toolchain to $MBPE_RUST_HOME (benchmark-only)..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --no-modify-path --profile minimal
        command -v cargo >/dev/null 2>&1 || { echo "  Error: Rust install failed" >&2; exit 1; }
    fi
fi

echo "=== Running benchmarks on: $CORPUS ==="

echo "  Building _mbpe.so..."
pixi run mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/mbpe/_mbpe.so

echo "  1/5  Mojo native (pre-trained encode/decode only)..."
BPE_CORPUS="$CORPUS" pixi run mojo -I . benchmarks/bm_pretrained.mojo > "$RESULTS/native.json"

echo "  2/5  mbpe Python bindings..."
BPE_CORPUS="$CORPUS" PYTHONPATH=python-binding pixi run -e dev python benchmarks/benchmark_mbpe.py > "$RESULTS/mbpe.json"

echo "  3/5  tiktoken (Python)..."
BPE_CORPUS="$CORPUS" pixi run -e dev python benchmarks/benchmark_tiktoken.py > "$RESULTS/tiktoken.json"

echo "  4/5  tiktoken-rs..."
BPE_CORPUS="$CORPUS" cargo run --release --manifest-path benchmarks/benchmark_rust/Cargo.toml 2>/dev/null > "$RESULTS/tiktoken-rs.json"

echo "  5/5  Mojo training (GPT4, 4 vocab sizes)..."
BPE_CORPUS="$CORPUS" pixi run mojo -I . benchmarks/bm_train.mojo > "$RESULTS/training.json"

echo ""
echo "=== Updating README.md ==="
bash scripts/update_readme_benchmarks.sh

echo ""
echo "=== Done ==="
