"""Lightweight mbpe benchmark — no file deps, runs in ~2s.

Trains a tokenizer on a small inline corpus, then benchmarks
encode/decode on a short text.  No Rust, no tiktoken, no setup.

Usage:  python benchmarks/smoke_bench.py
        PYTHONPATH=python-binding python benchmarks/smoke_bench.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-binding"))
import mbpe


CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
]
TEXT = " ".join(CORPUS) * 100  # ~10 KB


def main():
    results = []

    for name, cls in [
        ("gpt2", mbpe.GPT2Tokenizer),
        ("gpt4", mbpe.GPT4Tokenizer),
        ("gpt4o", mbpe.GPT4oTokenizer),
    ]:
        tok = cls()
        t0 = time.perf_counter_ns()
        tok.train(CORPUS, 300)
        train_ms = (time.perf_counter_ns() - t0) / 1_000_000

        ids = tok.encode(TEXT)
        n_tokens = len(ids)
        n_bytes = len(TEXT.encode("utf-8"))

        # encode
        t0 = time.perf_counter_ns()
        for _ in range(5):
            tok.encode(TEXT)
        encode_ms = (time.perf_counter_ns() - t0) / 5_000_000

        # decode
        t0 = time.perf_counter_ns()
        for _ in range(5):
            tok.decode(ids)
        decode_ms = (time.perf_counter_ns() - t0) / 5_000_000

        results.append({
            "impl": "mbpe_py",
            "encoding": name,
            "corpus_bytes": n_bytes,
            "n_tokens": n_tokens,
            "train_ms": round(train_ms, 1),
            "encode_ms": round(encode_ms, 3),
            "decode_ms": round(decode_ms, 3),
        })

    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
