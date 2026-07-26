"""Shared benchmark helpers — parameterized by pre-tokenizer type.

Usage:  mojo -I . -D BPE_PT=N benchmarks/bm.mojo
  N=0: GPreTokenizer   N=1: GPT2Pretokenizer   N=2: GPT4Pretokenizer
"""

from tokenizer import BPETokenizer
from pretokenizer import PreTokenizer
from std.pathlib import Path
from std.time import perf_counter_ns



struct Timer:
    var _start: UInt

    def __init__(out self):
        self._start = 0

    def start(mut self):
        self._start = perf_counter_ns()

    def elapsed_ns(self) -> Int:
        return Int(perf_counter_ns() - self._start)


def min_ns(times: List[Int]) -> Int:
    var best = times[0]
    for i in range(1, len(times)):
        if times[i] < best:
            best = times[i]
    return best


def fmt_ns(ns: Int) -> String:
    if ns < 1000:
        return String(ns) + " ns"
    elif ns < 1_000_000:
        return String(ns // 1000) + "." + String((ns % 1000) // 100) + " us"
    elif ns < 1_000_000_000:
        return (
            String(ns // 1_000_000)
            + "."
            + String((ns % 1_000_000) // 100_000)
            + " ms"
        )
    else:
        return (
            String(ns // 1_000_000_000)
            + "."
            + String((ns % 1_000_000_000) // 100_000_000)
            + " s"
        )


def fmt_tok_s(tokens: Int, ns: Int) -> String:
    if ns == 0:
        return "N/A"
    var per_sec = tokens * 1_000_000_000 // ns
    if per_sec >= 1_000_000:
        return (
            String(per_sec // 1_000_000)
            + "."
            + String((per_sec % 1_000_000) // 100_000)
            + " M tok/s"
        )
    elif per_sec >= 1000:
        return (
            String(per_sec // 1000)
            + "."
            + String((per_sec % 1000) // 100)
            + " K tok/s"
        )
    else:
        return String(per_sec) + " tok/s"


def run[PT: PreTokenizer](label: String) raises:
    print("=" * 60)
    print("  Mojo  — " + label)
    print("=" * 60)

    var corpus = Path("benchmarks/corpus.txt").read_text()
    var n_bytes = corpus.byte_length()
    print("Corpus: " + String(n_bytes) + " bytes")

    var lines = corpus.split("\n")
    var train_corpus = List[String]()
    for line in lines:
        var s = String(line)
        if s.byte_length() > 0:
            train_corpus.append(s^)

    print("Training (vocab_size=500)...")
    var timer = Timer()
    timer.start()
    var tok = BPETokenizer[PT]()
    tok.train(train_corpus, 500)
    var train_ns = timer.elapsed_ns()
    print("  train: " + fmt_ns(train_ns))
    print("  vocab: " + String(len(tok)) + "  merges: " + String(len(tok.merges)))

    var ids = tok.encode(corpus)
    var num_tokens = len(ids)
    print("  tokens: " + String(num_tokens))

    print("\n── encode ──")
    var encode_times = List[Int]()
    var n_iters = 20
    _ = tok.encode(corpus)
    for _ in range(n_iters):
        timer.start()
        _ = tok.encode(corpus)
        encode_times.append(timer.elapsed_ns())
    var enc_best = min_ns(encode_times)
    print("  best: " + fmt_ns(enc_best) + "  " + fmt_tok_s(num_tokens, enc_best))

    print("\n── decode ──")
    var decode_times = List[Int]()
    _ = tok.decode(ids)
    for _ in range(n_iters):
        timer.start()
        _ = tok.decode(ids)
        decode_times.append(timer.elapsed_ns())
    var dec_best = min_ns(decode_times)
    print("  best: " + fmt_ns(dec_best) + "  " + fmt_tok_s(num_tokens, dec_best))

    print("\n" + "=" * 60)
