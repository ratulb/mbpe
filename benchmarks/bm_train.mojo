"""Training-only benchmark for cl100k_base (GPT4Pretokenizer).

Outputs JSON lines: one per vocab_size (500, 1000, 2000, 4000).

Usage: BPE_CORPUS=benchmarks/corpus_5MB.txt mojo -I . benchmarks/bm_train.mojo
"""

from benchmark import run
from bpe.pretokenizer import GPT4Pretokenizer, ByteMapping


def main() raises:
    run[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]("GPT4")
