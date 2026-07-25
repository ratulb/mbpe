"""Benchmark suite for simple_bpe BPETokenizer encode/decode speed."""

from tokenizer import BPETokenizer
from std.time import perf_counter_ns


# ── timing helpers ───────────────────────────────────────────────────────

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


def mean_ns(times: List[Int]) -> Int:
    var total: Int = 0
    for i in range(len(times)):
        total += times[i]
    return total // len(times)


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


# ── corpus ───────────────────────────────────────────────────────────────

def build_corpus() raises -> String:
    """Return a concatenated test corpus as a single String."""
    var sentences = List[String]()
    sentences.append(String("This is the Hugging Face Course."))
    sentences.append(String("This chapter is about tokenization."))
    sentences.append(String("This section shows several tokenizer algorithms."))
    sentences.append(String(
        "Hopefully, you will be able to understand how they are trained and"
        " generate tokens."
    ))
    # Repeat to make a larger corpus
    var parts = List[String]()
    for _ in range(500):
        for i in range(len(sentences)):
            parts.append(sentences[i].copy())
    return String(" ".join(parts))


# ── benchmark ────────────────────────────────────────────────────────────

def run_benchmark() raises:
    print("simple_bpe — Benchmark Suite")
    print("============================\n")

    # Build corpus and train
    print("Building corpus...")
    var corpus = build_corpus()
    print("Corpus size:", String(corpus.byte_length()), "bytes\n")

    print("Training BPETokenizer (vocab_size=500)...")
    var tok = BPETokenizer()
    var timer = Timer()
    timer.start()
    tok.train([corpus], 500)
    var train_ns = timer.elapsed_ns()
    print("  train: " + fmt_ns(train_ns) + "\n")

    # Encode benchmark
    print("Encoding corpus...")
    var ids = tok.encode(corpus)
    var num_tokens = len(ids)
    print("  tokens:", String(num_tokens))

    var encode_times = List[Int]()
    var n_iters = 20
    # Warmup
    _ = tok.encode(corpus)
    for _ in range(n_iters):
        timer.start()
        _ = tok.encode(corpus)
        encode_times.append(timer.elapsed_ns())

    var enc_best = min_ns(encode_times)
    var enc_avg = mean_ns(encode_times)
    print("  encode (" + String(n_iters) + " iters):")
    print("    best: " + fmt_ns(enc_best) + "  " + fmt_tok_s(num_tokens, enc_best))
    print("    mean: " + fmt_ns(enc_avg) + "  " + fmt_tok_s(num_tokens, enc_avg))

    # Decode benchmark
    var decode_times = List[Int]()
    # Warmup
    _ = tok.decode(ids)
    for _ in range(n_iters):
        timer.start()
        _ = tok.decode(ids)
        decode_times.append(timer.elapsed_ns())

    var dec_best = min_ns(decode_times)
    var dec_avg = mean_ns(decode_times)
    print("  decode (" + String(n_iters) + " iters):")
    print("    best: " + fmt_ns(dec_best) + "  " + fmt_tok_s(num_tokens, dec_best))
    print("    mean: " + fmt_ns(dec_avg) + "  " + fmt_tok_s(num_tokens, dec_avg))

    # Roundtrip
    var rt_times = List[Int]()
    for _ in range(n_iters):
        timer.start()
        var e = tok.encode(corpus)
        _ = tok.decode(e)
        rt_times.append(timer.elapsed_ns())

    var rt_best = min_ns(rt_times)
    var rt_avg = mean_ns(rt_times)
    print("  roundtrip (" + String(n_iters) + " iters):")
    print("    best: " + fmt_ns(rt_best) + "  " + fmt_tok_s(num_tokens * 2, rt_best))
    print("    mean: " + fmt_ns(rt_avg) + "  " + fmt_tok_s(num_tokens * 2, rt_avg))

    print("\n================================")
    print("  Benchmark complete!")
    print("================================")


def main() raises:
    run_benchmark()
