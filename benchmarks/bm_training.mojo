"""Quick training benchmark -- GPT4Pretokenizer only."""
# Usage: BPE_CORPUS=benchmarks/corpus_5MB.txt mojo -I . benchmarks/bm_training.mojo

from benchmark import run_one
from bpe.pretokenizer import GPT4Pretokenizer, ByteMapping
from std.pathlib import Path
from std.os import getenv


def main() raises:
    var corpus_path = getenv("BPE_CORPUS", "benchmarks/corpus.txt")
    var full_text = Path(corpus_path).read_text()
    var n_bytes = full_text.byte_length()

    var lines = full_text.split("\n")
    var corpus = List[String]()
    for line in lines:
        var s = String(line)
        if s.byte_length() > 0:
            corpus.append(s^)

    var vocab_sizes: List[Int] = [500, 1000, 2000, 4000]
    run_one[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]("GPT4", corpus, full_text, n_bytes, vocab_sizes)
