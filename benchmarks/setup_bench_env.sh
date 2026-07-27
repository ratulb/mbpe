#!/usr/bin/env bash
# Set up benchmarking dependencies:
#   1. Rust via rustup (if missing)
#   2. Python venv at /tmp/mbpe-bench-venv/ with tiktoken
#
# Must be run under pixi (uses pixi run python for venv creation).
# Sources: outputs VENV_PYTHON path for consumers.
# Usage:  source benchmarks/setup_bench_env.sh

set -euo pipefail

VENV_DIR="/tmp/mbpe-bench-venv"

echo "── Setup: Rust ──"

# Install rustup if cargo isn't available
if ! command -v cargo &>/dev/null; then
    echo "  Installing rustup (latest stable)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -1
    source "$HOME/.cargo/env"
else
    echo "  Rust already installed: $(cargo --version)"
fi

export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$(pixi run which gcc 2>/dev/null || which gcc 2>/dev/null || true)"
if [ -z "${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER}" ]; then
    echo "  WARNING: no C compiler found (gcc). Rust builds may fail."
fi

echo ""
echo "── Setup: Python venv (${VENV_DIR}) ──"

if [ ! -d "$VENV_DIR" ]; then
    pixi run python -m venv "$VENV_DIR"
    echo "  Created venv"
fi

# Install/upgrade tiktoken
"$VENV_DIR/bin/pip" install -q --upgrade pip 2>/dev/null
"$VENV_DIR/bin/pip" install -q tiktoken 2>&1 | tail -1
echo "  tiktoken: $("$VENV_DIR/bin/python" -c 'import tiktoken; print(tiktoken.__version__)')"

export VENV_PYTHON="$VENV_DIR/bin/python"
echo ""
echo "── Setup complete ──"
