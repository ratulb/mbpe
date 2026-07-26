"""Benchmark entry point — runs all 3 pre-tokenizer variants.

Usage:  mojo -I . benchmarks/bm.mojo
        BPE_CORPUS=/path/to/corpus.txt mojo -I . benchmarks/bm.mojo
"""

from benchmark import run_all


def main() raises:
    run_all()
