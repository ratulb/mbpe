"""Native Mojo encoder-only benchmark — loads pre-built .tiktoken vocabs.
No training. Outputs JSON lines for each encoding (gpt2, cl100k, o200k).

Usage:  mojo -I . benchmarks/bm_pretrained.mojo
"""

from bpe.tokenizer import BPETokenizer
from bpe.pretokenizer import (
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


def bench_pretrained[
    PT: PreTokenizer,
](label: String, corpus: String, n_bytes: Int) raises:
    var tok = BPETokenizer[PT]()
    tok.load_tiktoken("data/" + label + ".tiktoken")
    var ids = tok.encode(corpus)
    var num_tokens = len(ids)

    # Encode — best of 3
    var encode_times = List[Int]()
    _ = tok.encode(corpus)
    for _ in range(3):
        var t0 = perf_counter_ns()
        _ = tok.encode(corpus)
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

    print(
        '{"impl":"mojo_native","encoding":"'
        + label
        + '","corpus_bytes":'
        + String(n_bytes)
        + ',"n_tokens":'
        + String(num_tokens)
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


def main() raises:
    var corpus_path = getenv("BPE_CORPUS", "benchmarks/corpus.txt")
    var corpus = Path(corpus_path).read_text()
    var n_bytes = corpus.byte_length()

    bench_pretrained[GPT2Pretokenizer]("gpt2", corpus, n_bytes)
    bench_pretrained[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]("cl100k", corpus, n_bytes)
    bench_pretrained[GPT4Pretokenizer[ByteMapping.SHUFFLED]]("o200k", corpus, n_bytes)
