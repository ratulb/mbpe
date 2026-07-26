#!/usr/bin/env python3
"""Collate benchmark JSON results into comparison tables.

Reads Mojo JSON (one line per (variant, vocab_size) combo),
Python tiktoken JSON (one line per encoding), and
Rust tiktoken-rs JSON (one line per encoding) from files,
then prints markdown comparison tables.

Usage:  python benchmarks/collate.py <mojo.json> <tiktoken.json> <tiktoken-rs.json>
"""

import json
import sys


def load_json_lines(path):
    """Load a JSON-lines file into a list of dicts. Returns [] on error/missing."""
    results = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip malformed lines
    except (FileNotFoundError, IOError):
        pass
    return results


def val(d, key, default="—"):
    """Safely extract a numeric value, formatting if needed."""
    v = d.get(key, default)
    if v is None or v == "N/A" or v == -1:
        return "—"
    return v


def fmt_ms(val):
    if val == "—":
        return val
    try:
        return f"{float(val):.1f}"
    except (ValueError, TypeError):
        return str(val)


def fmt_mtok(val):
    if val == "—":
        return val
    try:
        return f"{float(val):.1f}"
    except (ValueError, TypeError):
        return str(val)


def fmt_merges(val):
    if val == "—":
        return val
    try:
        return f"{float(val):.0f}"
    except (ValueError, TypeError):
        return str(val)


def make_mojo_table(mojo_rows):
    """Create a comparison table from Mojo results."""
    lines = []
    lines.append("### Mojo Pipeline (training + encode + decode)")
    lines.append("")
    lines.append("| Variant | Vocab | Merges | Train (ms) | Merges/s | Encode (ms) | Encode (M tok/s) | Decode (ms) | Decode (M tok/s) |")
    lines.append("|---------|-------|--------|-----------|----------|-------------|-----------------|-------------|-----------------|")
    for r in mojo_rows:
        lines.append(
            f"| {r['variant']} "
            f"| {r['vocab_size']} "
            f"| {r['n_merges']} "
            f"| {fmt_ms(r['train_ms'])} "
            f"| {fmt_merges(r['train_merges_s'])} "
            f"| {fmt_ms(r['encode_ms'])} "
            f"| {fmt_mtok(r['encode_mtok_s'])} "
            f"| {fmt_ms(r['decode_ms'])} "
            f"| {fmt_mtok(r['decode_mtok_s'])} |"
        )
    return "\n".join(lines)


def make_tiktoken_table(tiktoken_rows, title, impl_name):
    """Create a comparison table for tiktoken results (Python or Rust)."""
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| Encoding | Tokens | Encode (ms) | Encode (M tok/s) | Decode (ms) | Decode (M tok/s) |")
    lines.append("|----------|--------|-------------|-----------------|-------------|-----------------|")
    for r in tiktoken_rows:
        lines.append(
            f"| {r['encoding']} "
            f"| {r['n_tokens']} "
            f"| {fmt_ms(r['encode_ms'])} "
            f"| {fmt_mtok(r['encode_mtok_s'])} "
            f"| {fmt_ms(r['decode_ms'])} "
            f"| {fmt_mtok(r['decode_mtok_s'])} |"
        )
    return "\n".join(lines)


def make_encode_comparison(mojo_rows, tiktoken_rows, impl_a, impl_b):
    """Side-by-side encode comparison: Mojo (all variants) vs tiktoken."""
    lines = []
    lines.append(f"### Encode Comparison: Mojo vs {impl_a} vs {impl_b}")
    lines.append("")
    lines.append("| Variant | Mojo (M tok/s) | " + impl_a + " (M tok/s) | " + impl_b + " (M tok/s) |")
    lines.append("|---------|----------------|----------------------|----------------------|")

    # Group by encoding family
    # Mojo GPT2 → compare with tiktoken gpt2, Mojo GPT4 → compare with tiktoken cl100k
    # Use the first entry per variant (vocab_size=500)
    mojo_by_var = {}
    for r in mojo_rows:
        vs = r['vocab_size']
        if vs == 500:
            mojo_by_var[r['variant']] = r

    tiktoken_a = {r['encoding']: r for r in tiktoken_rows[0]} if tiktoken_rows else {}
    tiktoken_b = {r['encoding']: r for r in tiktoken_rows[1]} if len(tiktoken_rows) > 1 else {}

    if not tiktoken_a:
        tiktoken_a = {r['encoding']: r for r in tiktoken_rows}

    comparisons = [
        ("GPT2", "gpt2", "gpt2"),
        ("GPT4", "cl100k", "cl100k"),
    ]

    for mojo_variant, enc_a_name, enc_b_name in comparisons:
        m = mojo_by_var.get(mojo_variant, {})
        a = {}
        b = {}
        if isinstance(tiktoken_rows, list) and len(tiktoken_rows) > 0:
            # Find the correct encoding
            for r in tiktoken_rows[0]:
                if r['encoding'] == enc_a_name:
                    a = r
                    break
        if isinstance(tiktoken_rows, list) and len(tiktoken_rows) > 1:
            for r in tiktoken_rows[1]:
                if r['encoding'] == enc_b_name:
                    b = r
                    break

        lines.append(
            f"| {mojo_variant} "
            f"| {fmt_mtok(val(m, 'encode_mtok_s'))} "
            f"| {fmt_mtok(val(a, 'encode_mtok_s'))} "
            f"| {fmt_mtok(val(b, 'encode_mtok_s'))} |"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    mojo_path = sys.argv[1]
    py_path = sys.argv[2]
    rs_path = sys.argv[3]

    mojo = load_json_lines(mojo_path)
    py_tiktoken = load_json_lines(py_path)
    rs_tiktoken = load_json_lines(rs_path)

    # Corpus size
    n_bytes = mojo[0]['corpus_bytes'] if mojo else 0
    mb = n_bytes / 1_048_576 if n_bytes > 0 else 0
    if mb >= 1:
        corpus_label = f"{mb:.1f} MB"
    else:
        corpus_label = f"{n_bytes // 1024} KB"

    print(f"# Benchmark Results — {corpus_label} corpus")
    print()

    # Mojo table
    print(make_mojo_table(mojo))
    print()

    # Python tiktoken table
    if py_tiktoken:
        print(make_tiktoken_table(py_tiktoken, "Python tiktoken", "tiktoken_py"))
        print()

    # Rust tiktoken-rs table
    if rs_tiktoken:
        print(make_tiktoken_table(rs_tiktoken, "Rust tiktoken-rs", "tiktoken_rs"))
        print()

    # Cross-comparison: encode/decode (available benchmarks)
    py_encs = {r['encoding']: r for r in py_tiktoken}
    rs_encs = {r['encoding']: r for r in rs_tiktoken}

    if py_encs or rs_encs:
        print("### Encode/Decode Cross-Comparison (vocab_size=500 for Mojo)")
        print()
        cols = ["Encoding", "Mojo enc", "Mojo dec"]
        if py_encs:
            cols += ["Py enc", "Py dec"]
        if rs_encs:
            cols += ["Rust enc", "Rust dec"]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["-" * max(len(c), 3) for c in cols]) + " |"
        print(header)
        print(sep)

        # Get Mojo GPT2 and GPT4 at vocab_size=500
        mojo_map = {}
        for r in mojo:
            if r['vocab_size'] == 500:
                mojo_map[r['variant']] = r

        for mojo_var, enc_name in [("GPT2", "gpt2"), ("GPT4", "cl100k")]:
            m = mojo_map.get(mojo_var, {})
            m_enc = fmt_mtok(val(m, 'encode_mtok_s'))
            m_dec = fmt_mtok(val(m, 'decode_mtok_s'))
            row = [enc_name, m_enc, m_dec]
            if py_encs:
                p = py_encs.get(enc_name, {})
                row += [fmt_mtok(val(p, 'encode_mtok_s')), fmt_mtok(val(p, 'decode_mtok_s'))]
            if rs_encs:
                r = rs_encs.get(enc_name, {})
                row += [fmt_mtok(val(r, 'encode_mtok_s')), fmt_mtok(val(r, 'decode_mtok_s'))]
            print("| " + " | ".join(row) + " |")
        print()

    # Scaling table: per-variant, per-vocab-size
    print("### Scaling: Encode throughput by Vocab Size")
    print()
    print("| Variant | Vocab=500 | Vocab=1000 | Vocab=2000 | Vocab=4000 |")
    print("|---------|-----------|------------|------------|------------|")
    for variant in ["GPre", "GPT2", "GPT4"]:
        row = []
        for vs in [500, 1000, 2000, 4000]:
            matches = [r for r in mojo if r['variant'] == variant and r['vocab_size'] == vs]
            if matches:
                row.append(f"{fmt_mtok(val(matches[0], 'encode_mtok_s'))}")
        if row:
            print(f"| {variant} | {' | '.join(row)} |")

    print()
    print("### Scaling: Training time by Vocab Size")
    print()
    print("| Variant | Vocab=500 | Vocab=1000 | Vocab=2000 | Vocab=4000 |")
    print("|---------|-----------|------------|------------|------------|")
    for variant in ["GPre", "GPT2", "GPT4"]:
        row = []
        for vs in [500, 1000, 2000, 4000]:
            matches = [r for r in mojo if r['variant'] == variant and r['vocab_size'] == vs]
            if matches:
                row.append(f"{fmt_ms(val(matches[0], 'train_ms'))}")
        if row:
            print(f"| {variant} | {' | '.join(row)} |")


if __name__ == "__main__":
    main()
