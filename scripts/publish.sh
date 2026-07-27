#!/usr/bin/env bash
# Build the mbpe wheel for PyPI.
# Prerequisites: pixi installed, pixi.lock up-to-date.
#
# Usage:
#   bash scripts/publish.sh          # build wheel (repair with auditwheel)
#   bash scripts/publish.sh upload   # build + upload to PyPI
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── Building _mbpe.so ──" >&2
pixi run mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/mbpe/_mbpe.so

echo "── Copying data files into package ──" >&2
mkdir -p python-binding/mbpe/data
cp data/*.tiktoken python-binding/mbpe/data/

echo "── Building wheel ──" >&2
pip install build wheel auditwheel 2>/dev/null || true
python -m build --wheel

echo "── Repairing wheel (bundle Mojo runtime .so files) ──" >&2
auditwheel repair dist/*.whl -w dist/ 2>&1

echo "" >&2
echo "Wheel ready: $(ls dist/*.whl 2>/dev/null | head -1)" >&2

if [ "${1:-}" = "upload" ]; then
    echo "── Uploading to PyPI ──" >&2
    pip install twine 2>/dev/null || true
    twine upload dist/*.whl
fi
