#!/usr/bin/env bash
# runner for simple_bpe benchmarks.
# Installs dependencies (tiktoken pip package, Rust toolchain) then runs each
# benchmark and prints a comparison table.
#
# Usage:  bash benchmarks/run.sh

set -euo pipefail
cd "$(dirname "$0")/.."
BMDIR="benchmarks"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  simple_bpe  —  Multi-language Benchmark Suite              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Install tiktoken (Python pip) ──────────────────────────────
echo ""
echo "── [1/3] Installing Python tiktoken ──"
if python -c "import tiktoken" 2>/dev/null; then
    echo "  tiktoken already installed"
else
    echo "  Installing tiktoken via pip..."
    pip install tiktoken 2>&1 | tail -1
fi

# ── 2. Install Rust (if needed) ───────────────────────────────────
echo ""
echo "── [2/3] Installing Rust toolchain ──"
if command -v rustc &>/dev/null; then
    echo "  Rust $(rustc --version) already installed"
else
    echo "  Installing rustup (Rust toolchain installer)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y 2>&1 | tail -3
    source "$HOME/.cargo/env"
    echo "  Rust $(rustc --version) installed"
fi

# ── 3. Build Rust benchmark ───────────────────────────────────────
echo ""
echo "── [3/3] Building Rust benchmark ──"
(cd "$BMDIR/benchmark_rust" && cargo build --release 2>&1 | tail -2)

# ═══════════════════════════════════════════════════════════════════
echo ""
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Running Benchmarks"
echo "═══════════════════════════════════════════════════════════════"

MOJO_RESULT=$BMDIR/mojo_result.txt
PY_RESULT=$BMDIR/py_result.txt
RS_RESULT=$BMDIR/rs_result.txt

# ── Select Mojo entry point based on BPE_PT ──────────────────────
case "${BPE_PT:-}" in
  gpt2) MOJO_ENTRY="bm_gpt2.mojo"; MOJO_LABEL="BPETokenizer[GPT2Pretokenizer]" ;;
  gpt4) MOJO_ENTRY="bm_gpt4.mojo"; MOJO_LABEL="BPETokenizer[GPT4Pretokenizer]" ;;
  *)    MOJO_ENTRY="bm_default.mojo"; MOJO_LABEL="BPETokenizer[GPreTokenizer]" ;;
esac

# ── Mojo ──────────────────────────────────────────────────────────
echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  1/3  Mojo  ($MOJO_LABEL)"
echo "───────────────────────────────────────────────────────────────"
mojo -I . "$BMDIR/$MOJO_ENTRY" 2>&1 | tee "$MOJO_RESULT"

# ── Python tiktoken ───────────────────────────────────────────────
echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  2/3  Python tiktoken"
echo "───────────────────────────────────────────────────────────────"
python "$BMDIR/benchmark_tiktoken.py" 2>&1 | tee "$PY_RESULT"

# ── Rust tiktoken-rs ──────────────────────────────────────────────
echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  3/3  Rust tiktoken-rs"
echo "───────────────────────────────────────────────────────────────"
"$BMDIR/benchmark_rust/target/release/benchmark_rust" 2>&1 | tee "$RS_RESULT"

# ═══════════════════════════════════════════════════════════════════
# Extract and compare results
# ═══════════════════════════════════════════════════════════════════
echo ""
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results Summary"
echo "═══════════════════════════════════════════════════════════════"

# Extract "best:" lines — each file has encode then decode.
# Format:   best: XX.X ms   Y.Y M tok/s
# Extract encode/decode speeds from "best:" lines (first = encode, second = decode)
mojo_best=$(awk '/best:/ {print $4, $5, $6}' "$MOJO_RESULT")
py_best=$(awk '/best:/ {print $4, $5, $6}' "$PY_RESULT")
rs_best=$(awk '/best:/ {print $4, $5, $6}' "$RS_RESULT")

mojo_enc=$(echo "$mojo_best" | sed -n '1p')
mojo_dec=$(echo "$mojo_best" | sed -n '2p')
py_enc=$(echo "$py_best" | sed -n '1p')
py_dec=$(echo "$py_best" | sed -n '2p')
rs_enc=$(echo "$rs_best" | sed -n '1p')
rs_dec=$(echo "$rs_best" | sed -n '2p')

printf "\n%-22s %-16s %-16s\n" "" "encode (best)" "decode (best)"
printf "%s\n" "──────────────────────────────────────────────────────"
printf "%-22s %-16s %-16s\n" "Mojo"                "$mojo_enc" "$mojo_dec"
printf "%-22s %-16s %-16s\n" "tiktoken (Python)"   "$py_enc" "$py_dec"
printf "%-22s %-16s %-16s\n" "tiktoken-rs (Rust)"  "$rs_enc" "$rs_dec"
echo ""
