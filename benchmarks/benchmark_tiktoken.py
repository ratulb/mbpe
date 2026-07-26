"""tiktoken (Python) encode/decode speed benchmark — GPT-2 + cl100k_base.

Outputs JSON lines for each (encoding, operation) combination.

Usage: python benchmarks/benchmark_tiktoken.py
"""

import json
import os
import time

# Try importing tiktoken, with a helpful message if missing
try:
    import tiktoken
except ImportError:
    print("tiktoken not installed. Run: uv pip install tiktoken")
    print("  or: python -m pip install tiktoken")
    raise


def bench_encoding(name: str, enc, text: str, n_bytes: int, n_iters: int = 20):
    # First encode to get token count
    t0 = time.perf_counter_ns()
    tokens = enc.encode(text)
    t1 = time.perf_counter_ns()
    num_tokens = len(tokens)
    first_encode_ms = (t1 - t0) / 1_000_000

    # Encode benchmark (best of n_iters after warmup)
    _ = enc.encode(text)
    encode_times: list[int] = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = enc.encode(text)
        t1 = time.perf_counter_ns()
        encode_times.append(t1 - t0)

    enc_best_ns = min(encode_times)
    enc_best_ms = enc_best_ns / 1_000_000
    enc_mtok_s = num_tokens / (enc_best_ns / 1_000_000_000) / 1_000_000

    # Decode benchmark (best of n_iters after warmup)
    _ = enc.decode(tokens)
    decode_times: list[int] = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = enc.decode(tokens)
        t1 = time.perf_counter_ns()
        decode_times.append(t1 - t0)

    dec_best_ns = min(decode_times)
    dec_best_ms = dec_best_ns / 1_000_000
    dec_mtok_s = num_tokens / (dec_best_ns / 1_000_000_000) / 1_000_000

    result = {
        "impl": "tiktoken_py",
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
    # Resolve corpus from env var, with fallback
    corpus_path = os.environ.get("BPE_CORPUS")
    if corpus_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(script_dir, "corpus.txt")
    with open(corpus_path, "r") as f:
        text = f.read()
    n_bytes = len(text.encode("utf-8"))

    results = []

    # GPT-2 (r50k_base)
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    results.append(bench_encoding("gpt2", enc_gpt2, text, n_bytes))

    # GPT-4 (cl100k_base)
    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    results.append(bench_encoding("cl100k", enc_cl100k, text, n_bytes))

    # Print JSON lines (one per encoding)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
