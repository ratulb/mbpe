#!/usr/bin/env python3
"""Build cross-size summary tables from all benchmark results."""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Load all result files
def load(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (FileNotFoundError, IOError):
        pass
    return rows


# Corpus sizes (KB for labels)
CORPORA = {
    "10KB": 10,
    "100KB": 100,
    "500KB": 500,
    "1MB": 1024,
    "2MB": 2048,
    "5MB": 5120,
}

print("# Benchmark Results Summary")
print()

# ── Encode comparison across sizes ──
print("## Encode Throughput (M tok/s) — Mojo GPT2 vs tiktoken")
print()
print("| Corpus | Mojo GPT2 (v=500) | Python gpt2 | Rust gpt2 | Mojo / Py |")
print("|--------|-------------------|-------------|-----------|-----------|")

for label, kb in sorted(CORPORA.items(), key=lambda x: x[1]):
    mojo = load(os.path.join(RESULTS_DIR, f"mojo_{label}.json"))
    py = load(os.path.join(RESULTS_DIR, f"py_{label}.json"))
    rs = load(os.path.join(RESULTS_DIR, f"rs_{label}.json"))

    mojo_gpt2 = next((r for r in mojo if r['variant'] == 'GPT2' and r['vocab_size'] == 500), {})
    py_gpt2 = next((r for r in py if r['encoding'] == 'gpt2'), {})
    rs_gpt2 = next((r for r in rs if r['encoding'] == 'gpt2'), {})

    m_enc = mojo_gpt2.get('encode_mtok_s', 0)
    p_enc = py_gpt2.get('encode_mtok_s', 0)
    r_enc = rs_gpt2.get('encode_mtok_s', 0)
    ratio = m_enc / p_enc if p_enc else 0

    print(f"| {label} | {m_enc:.1f} | {p_enc:.1f} | {r_enc:.1f} | {ratio:.1f}x |")

print()

# ── Decode comparison across sizes ──
print("## Decode Throughput (M tok/s) — Mojo GPT2 vs tiktoken")
print()
print("| Corpus | Mojo GPT2 (v=500) | Python gpt2 | Rust gpt2 | Mojo / Py |")
print("|--------|-------------------|-------------|-----------|-----------|")

for label, kb in sorted(CORPORA.items(), key=lambda x: x[1]):
    mojo = load(os.path.join(RESULTS_DIR, f"mojo_{label}.json"))
    py = load(os.path.join(RESULTS_DIR, f"py_{label}.json"))
    rs = load(os.path.join(RESULTS_DIR, f"rs_{label}.json"))

    mojo_gpt2 = next((r for r in mojo if r['variant'] == 'GPT2' and r['vocab_size'] == 500), {})
    py_gpt2 = next((r for r in py if r['encoding'] == 'gpt2'), {})
    rs_gpt2 = next((r for r in rs if r['encoding'] == 'gpt2'), {})

    m_dec = mojo_gpt2.get('decode_mtok_s', 0)
    p_dec = py_gpt2.get('decode_mtok_s', 0)
    r_dec = rs_gpt2.get('decode_mtok_s', 0)
    ratio = m_dec / p_dec if p_dec else 0

    print(f"| {label} | {m_dec:.1f} | {p_dec:.1f} | {r_dec:.1f} | {ratio:.1f}x |")

print()

# ── Training time scaling ──
print("## Training Time (ms, GPT2) — Scaling by Corpus & Vocab Size")
print()
print("| Corpus | Vocab=500 | Vocab=1000 | Vocab=2000 | Vocab=4000 |")
print("|--------|-----------|------------|------------|------------|")

for label, kb in sorted(CORPORA.items(), key=lambda x: x[1]):
    mojo = load(os.path.join(RESULTS_DIR, f"mojo_{label}.json"))
    row = []
    for vs in [500, 1000, 2000, 4000]:
        r = next((x for x in mojo if x['variant'] == 'GPT2' and x['vocab_size'] == vs), {})
        row.append(f"{r.get('train_ms', 0):.0f}")
    print(f"| {label} | {' | '.join(row)} |")

print()

# ── Training merges/s scaling ──
print("## Training Throughput (merges/s, GPT2) — Scaling by Corpus & Vocab Size")
print()
print("| Corpus | Vocab=500 | Vocab=1000 | Vocab=2000 | Vocab=4000 |")
print("|--------|-----------|------------|------------|------------|")

for label, kb in sorted(CORPORA.items(), key=lambda x: x[1]):
    mojo = load(os.path.join(RESULTS_DIR, f"mojo_{label}.json"))
    row = []
    for vs in [500, 1000, 2000, 4000]:
        r = next((x for x in mojo if x['variant'] == 'GPT2' and x['vocab_size'] == vs), {})
        row.append(f"{r.get('train_merges_s', 0):.0f}")
    print(f"| {label} | {' | '.join(row)} |")

print()

# ── Pre-tokenizer split comparison at 5MB ──
print("## Pre-tokenizer Speed (5 MB corpus)")
print()
mojo_5mb = load(os.path.join(RESULTS_DIR, "mojo_5MB.json"))

print("| Variant | Train (ms, v=500) | Encode (ms, v=500) | Encode (M tok/s) | Decode (ms, v=500) | Decode (M tok/s) |")
print("|---------|-------------------|--------------------|-----------------|--------------------|-----------------|")
for variant in ["GPT2", "GPT4"]:
    r = next((x for x in mojo_5mb if x['variant'] == variant and x['vocab_size'] == 500), {})
    print(
        f"| {variant} "
        f"| {r.get('train_ms', 0):.0f} "
        f"| {r.get('encode_ms', 0):.1f} "
        f"| {r.get('encode_mtok_s', 0):.1f} "
        f"| {r.get('decode_ms', 0):.1f} "
        f"| {r.get('decode_mtok_s', 0):.1f} |"
    )

print()
print("## Cross-Language Comparison (5 MB, Mojo GPT2 vs tiktoken)")
print()
py_5mb = load(os.path.join(RESULTS_DIR, "py_5MB.json"))
rs_5mb = load(os.path.join(RESULTS_DIR, "rs_5MB.json"))
m = next((x for x in mojo_5mb if x['variant'] == 'GPT2' and x['vocab_size'] == 500), {})
p = next((x for x in py_5mb if x['encoding'] == 'gpt2'), {})
r = next((x for x in rs_5mb if x['encoding'] == 'gpt2'), {})

print("| Metric | Mojo (ours) | Python tiktoken | Rust tiktoken-rs |")
print("|--------|------------|----------------|-----------------|")
print(f"| Encode (ms) | {m.get('encode_ms', 0):.1f} | {p.get('encode_ms', 0):.1f} | {r.get('encode_ms', 0):.1f} |")
print(f"| Encode (M tok/s) | {m.get('encode_mtok_s', 0):.1f} | {p.get('encode_mtok_s', 0):.1f} | {r.get('encode_mtok_s', 0):.1f} |")
print(f"| Decode (ms) | {m.get('decode_ms', 0):.1f} | {p.get('decode_ms', 0):.1f} | {r.get('decode_ms', 0):.1f} |")
print(f"| Decode (M tok/s) | {m.get('decode_mtok_s', 0):.1f} | {p.get('decode_mtok_s', 0):.1f} | {r.get('decode_mtok_s', 0):.1f} |")
print(f"| Train (ms) | {m.get('train_ms', 0):.0f} | N/A | N/A |")
print(f"| Train (merges/s) | {m.get('train_merges_s', 0):.0f} | N/A | N/A |")
