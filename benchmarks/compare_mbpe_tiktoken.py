"""Compare mbpe vs tiktoken encode/decode speed on a web-fetched corpus.

Paste into a Colab cell (prepend: `!pip install mbpe tiktoken`)
or run locally:  pip install mbpe tiktoken && python benchmarks/compare_mbpe_tiktoken.py
"""

import time
import urllib.request
import mbpe
import tiktoken

N_ITER = 5

URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
print(f"Downloading {URL}...")
text = urllib.request.urlopen(URL).read().decode("utf-8")
print(f"Corpus: {len(text):,} bytes ({len(text.split()):,} words)\n")

encodings = [
    ("gpt2",   mbpe.get_encoding("gpt2"),     tiktoken.get_encoding("gpt2")),
    ("cl100k", mbpe.get_encoding("cl100k"),   tiktoken.get_encoding("cl100k_base")),
    ("o200k",  mbpe.get_encoding("o200k"),    tiktoken.get_encoding("o200k_base")),
]

for label, m_tok, t_tok in encodings:
    ids_m = m_tok.encode(text)
    ids_t = t_tok.encode(text)
    print('mbpe produces same ids as tiktoken: ', ids_m == ids_t)

    # warmup
    _ = m_tok.decode(ids_m); _ = t_tok.decode(ids_t)

    t0 = time.perf_counter()
    for _ in range(N_ITER): m_tok.encode(text)
    m_enc = (time.perf_counter() - t0) / N_ITER

    t0 = time.perf_counter()
    for _ in range(N_ITER): m_tok.decode(ids_m)
    m_dec = (time.perf_counter() - t0) / N_ITER

    t0 = time.perf_counter()
    for _ in range(N_ITER): t_tok.encode(text)
    t_enc = (time.perf_counter() - t0) / N_ITER

    t0 = time.perf_counter()
    for _ in range(N_ITER): t_tok.decode(ids_t)
    t_dec = (time.perf_counter() - t0) / N_ITER

    print(f"── {label} ({len(ids_m):,} tokens) ──")
    print(f"  mbpe     encode {len(ids_m) / m_enc / 1e6:.1f}M tok/s  decode {len(ids_m) / m_dec / 1e6:.1f}M tok/s")
    print(f"  tiktoken encode {len(ids_t) / t_enc / 1e6:.1f}M tok/s  decode {len(ids_t) / t_dec / 1e6:.1f}M tok/s")
    print()
