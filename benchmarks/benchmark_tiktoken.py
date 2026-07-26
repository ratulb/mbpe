"""tiktoken (Python) encode/decode speed benchmark.

Usage: python benchmarks/benchmark_tiktoken.py
"""

import os
import time

# Try importing tiktoken, with a helpful message if missing
try:
    import tiktoken
except ImportError:
    print("tiktoken not installed. Run: uv pip install tiktoken")
    print("  or: python -m pip install tiktoken")
    raise


def fmt_tok_s(tokens: int, ns: int) -> str:
    if ns == 0:
        return "N/A"
    per_sec = int(tokens * 1_000_000_000 / ns)
    if per_sec >= 1_000_000:
        return f"{per_sec / 1_000_000:.1f} M tok/s"
    elif per_sec >= 1000:
        return f"{per_sec / 1000:.1f} K tok/s"
    else:
        return f"{per_sec} tok/s"


def fmt_ns(ns: int) -> str:
    if ns < 1000:
        return f"{ns} ns"
    elif ns < 1_000_000:
        return f"{ns / 1000:.1f} us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def main():
    print("=" * 60)
    print("  tiktoken (Python)  — GPT-2 BPE encode/decode benchmark")
    print("=" * 60)

    # Resolve corpus from env var, with fallback
    corpus_path = os.environ.get("BPE_CORPUS")
    if corpus_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(script_dir, "corpus.txt")
    with open(corpus_path, "r") as f:
        text = f.read()
    n_bytes = len(text.encode("utf-8"))
    print(f"\nCorpus: {n_bytes} bytes")

    # Load GPT-2 tokenizer (Rust native under the hood)
    enc = tiktoken.get_encoding("gpt2")
    print(f"  vocab: {enc.n_vocab}")

    # First encode to get token count
    t0 = time.perf_counter_ns()
    tokens = enc.encode(text)
    t1 = time.perf_counter_ns()
    num_tokens = len(tokens)
    print(f"  tokens: {num_tokens}  (first encode: {fmt_ns(t1 - t0)})")

    # Encode benchmark
    print("\n── encode ──")
    n_iters = 20
    _ = enc.encode(text)  # warmup
    encode_times: list[int] = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = enc.encode(text)
        t1 = time.perf_counter_ns()
        encode_times.append(t1 - t0)

    enc_best = min(encode_times)
    print(f"  best: {fmt_ns(enc_best)}  {fmt_tok_s(num_tokens, enc_best)}")

    # Decode benchmark
    print("\n── decode ──")
    _ = enc.decode(tokens)  # warmup
    decode_times: list[int] = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _ = enc.decode(tokens)
        t1 = time.perf_counter_ns()
        decode_times.append(t1 - t0)

    dec_best = min(decode_times)
    print(f"  best: {fmt_ns(dec_best)}  {fmt_tok_s(num_tokens, dec_best)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
