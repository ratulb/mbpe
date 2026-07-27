"""mbpe Python bindings encode/decode speed benchmark (quick version).

Same as benchmark_mbpe.py but with n_iters=3 for faster results.

Usage: python benchmarks/benchmark_mbpe_quick.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mbpe


def bench_encoding(name: str, enc, text: str, n_bytes: int, n_iters: int = 3):
    t0 = time.perf_counter_ns()
    tokens = enc.encode(text)
    t1 = time.perf_counter_ns()
    num_tokens = len(tokens)
    first_encode_ms = (t1 - t0) / 1_000_000

    _ = enc.encode(text)
    encode_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = enc.encode(text)
        t1 = time.perf_counter_ns()
        encode_times.append(t1 - t0)
    enc_best_ns = min(encode_times)
    enc_best_ms = enc_best_ns / 1_000_000
    enc_mtok_s = num_tokens / (enc_best_ns / 1_000_000_000) / 1_000_000

    _ = enc.decode(tokens)
    decode_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = enc.decode(tokens)
        t1 = time.perf_counter_ns()
        decode_times.append(t1 - t0)
    dec_best_ns = min(decode_times)
    dec_best_ms = dec_best_ns / 1_000_000
    dec_mtok_s = num_tokens / (dec_best_ns / 1_000_000_000) / 1_000_000

    result = {
        "impl": "mbpe_py",
        "encoding": name,
        "corpus_bytes": n_bytes,
        "n_tokens": num_tokens,
        "first_encode_ms": round(first_encode_ms, 2),
        "encode_ms": round(enc_best_ms, 2),
        "encode_mtok_s": round(enc_mtok_s, 2),
        "decode_ms": round(dec_best_ms, 2),
        "decode_mtok_s": round(dec_mtok_s, 2),
    }
    return result


def main():
    corpus_path = os.environ.get("BPE_CORPUS")
    if corpus_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(script_dir, "corpus_1MB.txt")
    with open(corpus_path) as f:
        text = f.read()
    n_bytes = len(text.encode("utf-8"))

    results = []
    for name in ["gpt2", "cl100k", "o200k"]:
        enc = mbpe.get_encoding(name)
        results.append(bench_encoding(name, enc, text, n_bytes))

    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
