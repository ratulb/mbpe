#!/usr/bin/env bash
# Local conda-recipe validation with rattler-build (linux-64 only).
# Builds the recipe end-to-end in an isolated environment and runs its test phase —
# the same validation modular-community CI performs on the x86_64-linux leg.
#
# Usage:  bash rattler_build_test.sh [recipe_path]
#   Default recipe: conda.recipe/recipe.yaml (local dev form, source.path: ..)
#   CI form (e.g. the fork recipe): bash rattler_build_test.sh ../modular-community/recipes/mbpe/recipe.yaml
#
# First run installs rattler-build globally via pixi (pixi global install rattler-build).
# Build artifacts land in <recipe_dir>/output/ (gitignored in both repos).
set -euo pipefail

cd "$(dirname "$0")"

RECIPE="${1:-${RATTLER_RECIPE:-conda.recipe/recipe.yaml}}"
CHANNELS=(-c conda-forge -c https://conda.modular.com/max -c https://repo.prefix.dev/modular-community)

# ── Ensure rattler-build exists ────────────────────────────────
if ! command -v rattler-build >/dev/null 2>&1; then
    echo "  rattler-build not found — installing globally via pixi (first run only)..."
    pixi global install rattler-build
    command -v rattler-build >/dev/null 2>&1 || { echo "  Error: rattler-build install failed" >&2; exit 1; }
fi

echo "=== Local conda-recipe validation: $RECIPE ==="
echo "  Channels: conda-forge, conda.modular.com/max, repo.prefix.dev/modular-community"

# ── Build from the recipe's directory ──────────────────────────
# Keeps output/ (gitignored) next to the recipe and makes the local dev form's
# relative source path (../) resolve correctly.
RECDIR="$(cd "$(dirname "$RECIPE")" && pwd)"
RECBASE="$(basename "$RECIPE")"

if ! ( cd "$RECDIR" && rattler-build build --recipe "$RECBASE" "${CHANNELS[@]}" ); then
    echo ""
    echo "  Build FAILED. Interactive debug:  cd $RECDIR && rattler-build debug shell --recipe $RECBASE" >&2
    exit 1
fi

echo ""
echo "=== Validation passed ==="
echo "  1. recipe schema, pin_compatible('mojo-compiler'), match specs"
echo "  2. source fetch (git+rev in the CI form, local path in the dev form)"
echo "  3. mojo-compiler =1.0.0b2 resolution from conda.modular.com/max"
echo "  4. build script: mojo precompile bpe -o \$PREFIX/lib/mojo/bpe.mojoc,"
echo "     data/*.tiktoken copy, activate.d/deactivate.d MBPE_DATA_DIR hooks"
echo "  5. .conda archive packaged"
echo "  6. test phase: mojo run test_import.mojo in an isolated env (package loads)"
echo "  7. license_file resolved from the source tree"
echo ""
echo "  Note: linux-64 only — the arm64 and macos CI legs are covered by"
echo "  modular-community's build-all matrix (needs the 'OK to test' label)."
