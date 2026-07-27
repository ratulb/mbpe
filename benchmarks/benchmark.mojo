"""Shared benchmark helpers — split-only, full pipeline, multi-variant.
Outputs JSON lines for each (variant, vocab_size) combination.

Usage:  mojo -I . benchmarks/bm.mojo
"""

from bpe.tokenizer import BPETokenizer
from bpe.pretokenizer import (
    GPreTokenizer,
    GPT2Pretokenizer,
    GPT4Pretokenizer,
    PreTokenizer,
    ByteMapping,
)
from std.pathlib import Path
from std.os import getenv
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


def fmt_mtok_s(tokens: Int, ns: Int) -> Float64:
    if ns == 0:
        return 0.0
    return Float64(tokens) / (Float64(ns) / 1_000_000_000.0) / 1_000_000.0


def ns_to_ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


# ── Pre-tokenization split benchmark ──────────────────────────────

@always_inline
def measure_split_ns[PT: PreTokenizer](text: String) raises -> Tuple[Int, Int]:
    var pt = PT()
    var t0 = perf_counter_ns()
    var words = pt.split(text)
    var ns = Int(perf_counter_ns() - t0)
    return (ns, len(words))


# ── Full pipeline benchmark ───────────────────────────────────────

def run_one[PT: PreTokenizer](
    label: String,
    corpus: Span[String, _],
    full_text: String,
    n_bytes: Int,
    vocab_sizes: List[Int],
) raises:
    for vs in range(len(vocab_sizes)):
        var vsize = vocab_sizes[vs]

        # Training (single run for speed on large corpora)
        var train_times = List[Int]()
        for _ in range(1):
            var t = BPETokenizer[PT]()
            var t0 = perf_counter_ns()
            t.train(corpus, vsize)
            train_times.append(Int(perf_counter_ns() - t0))
        var best_train_ns = min_ns(train_times)

        var tok = BPETokenizer[PT]()
        tok.train(corpus, vsize)
        var ids = tok.encode(full_text)
        var num_tokens = len(ids)

        # Encode — best of 3
        var encode_times = List[Int]()
        _ = tok.encode(full_text)
        for _ in range(3):
            var t0 = perf_counter_ns()
            _ = tok.encode(full_text)
            encode_times.append(Int(perf_counter_ns() - t0))
        var best_encode_ns = min_ns(encode_times)

        # Decode — best of 3
        var decode_times = List[Int]()
        _ = tok.decode(ids)
        for _ in range(3):
            var t0 = perf_counter_ns()
            _ = tok.decode(ids)
            decode_times.append(Int(perf_counter_ns() - t0))
        var best_decode_ns = min_ns(decode_times)

        var n_merges = len(tok.merges)
        var n_vocab = len(tok)
        var train_merges_s = 0
        if best_train_ns > 0:
            train_merges_s = n_merges * 1_000_000_000 // best_train_ns

        print(
            '{"impl":"mojo","variant":"'
            + label
            + '","corpus_bytes":'
            + String(n_bytes)
            + ',"vocab_size":'
            + String(vsize)
            + ',"n_vocab":'
            + String(n_vocab)
            + ',"n_merges":'
            + String(n_merges)
            + ',"n_tokens":'
            + String(num_tokens)
            + ',"train_ms":'
            + String(ns_to_ms(best_train_ns))
            + ',"train_merges_s":'
            + String(train_merges_s)
            + ',"encode_ms":'
            + String(ns_to_ms(best_encode_ns))
            + ',"encode_mtok_s":'
            + String(fmt_mtok_s(num_tokens, best_encode_ns))
            + ',"decode_ms":'
            + String(ns_to_ms(best_decode_ns))
            + ',"decode_mtok_s":'
            + String(fmt_mtok_s(num_tokens, best_decode_ns))
            + "}"
        )


# ── Unified runner ────────────────────────────────────────────────

def run_all() raises:
    var corpus_path = getenv("BPE_CORPUS", "benchmarks/corpus.txt")
    var full_text = Path(corpus_path).read_text()
    var n_bytes = full_text.byte_length()

    # Build training corpus (non-empty lines)
    var lines = full_text.split("\n")
    var corpus = List[String]()
    for line in lines:
        var s = String(line)
        if s.byte_length() > 0:
            corpus.append(s^)

    var vocab_sizes: List[Int] = [500, 1000, 2000, 4000]

    run_one[GPreTokenizer]("GPre", corpus, full_text, n_bytes, vocab_sizes)
    run_one[GPT2Pretokenizer]("GPT2", corpus, full_text, n_bytes, vocab_sizes)
    run_one[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]("GPT4", corpus, full_text, n_bytes, vocab_sizes)


# ── Legacy single-variant entry point ─────────────────────────────

def run[PT: PreTokenizer](label: String) raises:
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
    run_one[PT](label, corpus, full_text, n_bytes, vocab_sizes)
