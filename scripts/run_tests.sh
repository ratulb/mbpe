#!/usr/bin/env bash
# One-command test suite: builds the Python binding from scratch, then runs
# every Python and Mojo test in the repo (same order as CI).
#
# Usage:  bash scripts/run_tests.sh [--skip-cov]
#
# The shared lib is always deleted first, so tests can never run against a
# stale binary.
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_COV=0
for arg in "$@"; do
    case "$arg" in
        --skip-cov) SKIP_COV=1 ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: bash scripts/run_tests.sh [--skip-cov]" >&2
            exit 1
            ;;
    esac
done

SO="python-binding/mbpe/_mbpe.so"

echo "── Removing stale $SO..."
rm -f "$SO"

echo "── Building $SO..."
pixi run mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o "$SO"

echo "── Python tests (pytest, all 4 tokenizer variants)..."
pixi run --environment dev python -m pytest tests/python/ -v --tb=short

if [ "$SKIP_COV" -eq 0 ]; then
    echo "── Python coverage..."
    pixi run --environment dev python -m pytest tests/python/ --cov=mbpe --cov-report=term-missing
fi

echo "── Mojo tests (main suite)..."
pixi run mojo main.mojo

echo "── Mojo tests (tokenizer)..."
pixi run mojo -I . tests/test_tokenizer.mojo

echo "── Mojo tests (exhaustive)..."
pixi run mojo -I . tests/exhaustive_tokenizer.mojo

echo ""
echo "All tests passed."
