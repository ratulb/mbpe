#!/usr/bin/env python3
"""Collate benchmark JSON results into a structured markdown report.

Reads Mojo JSON (one line per (variant, vocab_size) combo),
Python tiktoken JSON (one line per encoding),
Rust tiktoken-rs JSON (one line per encoding), and
mbpe Python bindings JSON (one line per encoding) from files,
then prints a markdown report with executive summary + comparison tables.

Usage:  python benchmarks/collate.py <mojo.json> <tiktoken.json> <tiktoken-rs.json> <mbpe.json>
"""

import datetime
import json
import platform
import subprocess
import sys


# ── Hardware detection ──────────────────────────────────────────

def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def _cpu_model():
    out = _run(["lscpu"])
    for line in out.split("\n"):
        if "Model name" in line:
            return line.split(":", 1)[-1].strip()
    out = _run(["cat", "/proc/cpuinfo"])
    for line in out.split("\n"):
        if line.startswith("model name"):
            return line.split(":", 1)[-1].strip()
    return platform.processor() or "N/A"


def _cpu_cores():
    out = _run(["nproc"])
    try:
        return int(out.strip())
    except (ValueError, TypeError):
        return "N/A"


def _ram():
    out = _run(["free", "-h"])
    for line in out.split("\n"):
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return "N/A"


def _os():
    out = _run(["cat", "/etc/os-release"])
    for line in out.split("\n"):
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[-1].strip().strip('"')
    return platform.system() + " " + platform.release()


def _mojo_version():
    return _run(["mojo", "--version"])


def _python_version():
    return _run(["pixi", "run", "python", "--version"]) or sys.version.split()[0]


def _rust_version():
    return _run(["rustc", "--version"])


def _tiktoken_version():
    try:
        import tiktoken
        return tiktoken.__version__
    except ImportError:
        return "N/A"


def make_hardware_table():
    lines = []
    lines.append("### Benchmark Environment")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Date | {datetime.date.today()} |")
    lines.append(f"| CPU | {_cpu_model()} |")
    lines.append(f"| Cores | {_cpu_cores()} (logical) |")
    lines.append(f"| RAM | {_ram()} |")
    lines.append(f"| OS | {_os()} |")
    lines.append(f"| Mojo | {_mojo_version()} |")
    lines.append(f"| Python (pixi) | {_python_version()} |")
    lines.append(f"| Rust | {_rust_version()} |")
    lines.append(f"| tiktoken | {_tiktoken_version()} |")
    lines.append("")
    return "\n".join(lines)


# ── JSON loading ────────────────────────────────────────────────

def load_json_lines(path):
    results = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        if isinstance(d, dict) and "impl" in d:
                            results.append(d)
                    except json.JSONDecodeError:
                        pass
    except (FileNotFoundError, IOError):
        pass
    return results


def get(d, key, default="—"):
    v = d.get(key, default)
    if v is None or v == "N/A" or v == -1:
        return "—"
    return v


def fmt_ms(v):
    if v == "—":
        return v
    try:
        return f"{float(v):.1f}"
    except (ValueError, TypeError):
        return str(v)


def fmt_mtok(v):
    if v == "—":
        return v
    try:
        return f"{float(v):.1f}"
    except (ValueError, TypeError):
        return str(v)


def fmt_mbs(v):
    if v == "—":
        return v
    try:
        return f"{float(v):.1f}"
    except (ValueError, TypeError):
        return str(v)


def mb_per_s(n_bytes, time_ms):
    if time_ms == "—" or not time_ms:
        return "—"
    try:
        t = float(time_ms)
        return n_bytes * 1000 / (t * 1_048_576)
    except (ValueError, ZeroDivisionError, TypeError):
        return "—"


# ── Executive Summary ───────────────────────────────────────────

def make_summary(mbpe_py, py_tiktoken, n_bytes):
    lines = []

    mbpe_enc_best = None
    mbpe_dec_best = None
    for r in mbpe_py:
        enc = get(r, 'encode_mtok_s')
        dec = get(r, 'decode_mtok_s')
        if enc != "—":
            try:
                if mbpe_enc_best is None or float(enc) > mbpe_enc_best[0]:
                    mbpe_enc_best = (float(enc), r['encoding'])
            except (ValueError, TypeError):
                pass
        if dec != "—":
            try:
                if mbpe_dec_best is None or float(dec) > mbpe_dec_best[0]:
                    mbpe_dec_best = (float(dec), r['encoding'])
            except (ValueError, TypeError):
                pass

    py_enc_best = None
    py_dec_best = None
    for r in py_tiktoken:
        enc = get(r, 'encode_mtok_s')
        dec = get(r, 'decode_mtok_s')
        if enc != "—":
            try:
                if py_enc_best is None or float(enc) > py_enc_best[0]:
                    py_enc_best = (float(enc), r['encoding'])
            except (ValueError, TypeError):
                pass
        if dec != "—":
            try:
                if py_dec_best is None or float(dec) > py_dec_best[0]:
                    py_dec_best = (float(dec), r['encoding'])
            except (ValueError, TypeError):
                pass

    bullets = []

    if mbpe_enc_best:
        bullets.append(f"⚡ mbpe Python bindings encode at {mbpe_enc_best[0]:.1f} M tok/s ({mbpe_enc_best[1]})")

    if mbpe_dec_best:
        bullets.append(f"⚡ mbpe Python bindings decode at {mbpe_dec_best[0]:.1f} M tok/s ({mbpe_dec_best[1]})")

    if mbpe_enc_best and py_enc_best:
        ratio = mbpe_enc_best[0] / py_enc_best[0]
        bullets.append(f"🏁 {ratio:.1f}× encode vs Python tiktoken")

    if mbpe_dec_best and py_dec_best:
        ratio = mbpe_dec_best[0] / py_dec_best[0]
        bullets.append(f"🏁 {ratio:.1f}× decode vs Python tiktoken")

    bullets.append("📦 Built-in support for .tiktoken vocabularies (gpt2 ~50K, cl100k ~100K, o200k ~200K)")

    lines.append("> **Highlights**")
    for b in bullets:
        lines.append(f">")
        lines.append(f"> {b}")
    lines.append(">")
    lines.append("")
    return "\n".join(lines)


# ── Combined Comparison Table ───────────────────────────────────

VARIANT_LABELS = {
    "GPT2": "GPT2Pretokenizer (r50k_base)",
    "GPT4": "GPT4Pretokenizer (cl100k_base)",
}


def make_comparison_table(mbpe_py, py_tiktoken, rs_tiktoken, mojo, mojo_native, n_bytes):
    lines = []
    mb = n_bytes / 1_048_576 if n_bytes > 0 else 0
    corpus_label = f"{mb:.1f} MB" if mb >= 1 else f"{n_bytes // 1024} KB"
    lines.append(f"### Encode/Decode Throughput — {corpus_label} corpus")
    lines.append("")
    lines.append("")

    header = (
        "| Encoding | Implementation | Tokens | Encode (ms) | "
        "Encode (M tok/s) | Encode (MB/s) | Decode (ms) | "
        "Decode (M tok/s) | Decode (MB/s) |"
    )
    sep = "|" + "|".join(["---"] * 9) + "|"
    lines.append(header)
    lines.append(sep)

    mbpe_by_enc = {r['encoding']: r for r in mbpe_py}
    py_by_enc = {r['encoding']: r for r in py_tiktoken}
    rs_by_enc = {r['encoding']: r for r in rs_tiktoken}
    native_by_enc = {r['encoding']: r for r in mojo_native}

    encodings = ["gpt2", "cl100k", "o200k"]

    for enc in encodings:
        impls = []
        if enc in native_by_enc:
            impls.append(("Mojo native", native_by_enc[enc]))
        if enc in mbpe_by_enc:
            impls.append(("mbpe (Python)", mbpe_by_enc[enc]))
        if enc in py_by_enc:
            impls.append(("tiktoken (Python)", py_by_enc[enc]))
        if enc in rs_by_enc:
            impls.append(("tiktoken-rs", rs_by_enc[enc]))
        if not impls:
            continue

        for label, r in impls:
            tok = get(r, 'n_tokens')
            e_ms = get(r, 'encode_ms')
            e_mtok = get(r, 'encode_mtok_s')
            e_mbs = mb_per_s(n_bytes, e_ms)
            d_ms = get(r, 'decode_ms')
            d_mtok = get(r, 'decode_mtok_s')
            d_mbs = mb_per_s(n_bytes, d_ms)
            display = f"**{label}**" if label.startswith("mbpe") else label
            lines.append(
                f"| {enc} "
                f"| {display} "
                f"| {tok} "
                f"| {fmt_ms(e_ms)} "
                f"| {fmt_mtok(e_mtok)} "
                f"| {fmt_mbs(e_mbs)} "
                f"| {fmt_ms(d_ms)} "
                f"| {fmt_mtok(d_mtok)} "
                f"| {fmt_mbs(d_mbs)} |"
            )

    lines.append("")
    notes = []

    gpt2_tokens = {}
    for label, lookup in [("Mojo native", native_by_enc), ("mbpe (Python)", mbpe_by_enc), ("tiktoken (Python)", py_by_enc), ("tiktoken-rs", rs_by_enc)]:
        if "gpt2" in lookup:
            gpt2_tokens[label] = lookup["gpt2"].get('n_tokens')
    token_vals = {v for v in gpt2_tokens.values() if v and v != "—"}
    if len(token_vals) > 1:
        notes.append(
            "tiktoken-rs gpt2 token count differs from Python — likely a "
            "pre-tokenizer regex version mismatch between the tiktoken-rs crate "
            "and OpenAI/tiktoken."
        )

    for i, note in enumerate(notes):
        lines.append(f"_{i+1}. {note}_")
        lines.append("")

    return "\n".join(lines)


# ── Native Mojo Pipeline ─────────────────────────────────────────

def make_mojo_table(mojo_rows):
    lines = []
    lines.append("### Mojo Native Pipeline — Training + Encode + Decode")
    lines.append("")
    lines.append("| Variant | Vocab | Merges | Train (ms) | Merges/s | Encode (ms) | Encode (M tok/s) | Decode (ms) | Decode (M tok/s) |")
    lines.append("|---------|-------|--------|-----------|----------|-------------|-----------------|-------------|-----------------|")
    for r in mojo_rows:
        var_label = VARIANT_LABELS.get(r['variant'], r['variant'])
        tr = get(r, 'train_merges_s', 0)
        merges_s = f"{float(tr):.0f}" if tr != "—" else "—"
        lines.append(
            f"| {var_label} "
            f"| {r['vocab_size']} "
            f"| {r['n_merges']} "
            f"| {fmt_ms(get(r, 'train_ms'))} "
            f"| {merges_s} "
            f"| {fmt_ms(get(r, 'encode_ms'))} "
            f"| {fmt_mtok(get(r, 'encode_mtok_s'))} "
            f"| {fmt_ms(get(r, 'decode_ms'))} "
            f"| {fmt_mtok(get(r, 'decode_mtok_s'))} |"
        )
    return "\n".join(lines)


# ── Scaling Tables ───────────────────────────────────────────────

def make_scaling_tables(mojo_rows):
    lines = []
    lines.append("### Mojo Throughput Scaling by Vocab Size")
    lines.append("")
    lines.append("*Encode speed and training time both decrease as vocabulary grows "
                 "(longer merge chains). These numbers use self-trained vocabularies "
                 "and are not directly comparable to 50K+ tokenizers above.*")
    lines.append("")

    lines.append("**Encode throughput (M tok/s)**")
    lines.append("")
    lines.append("| Variant | Vocab=500 | Vocab=1000 | Vocab=2000 | Vocab=4000 |")
    lines.append("|---------|-----------|------------|------------|------------|")
    for variant in ["GPT2", "GPT4"]:
        var_label = VARIANT_LABELS.get(variant, variant)
        row = []
        for vs in [500, 1000, 2000, 4000]:
            matches = [r for r in mojo_rows if r['variant'] == variant and r['vocab_size'] == vs]
            if matches:
                row.append(f"{fmt_mtok(get(matches[0], 'encode_mtok_s'))}")
        if row:
            lines.append(f"| {var_label} | {' | '.join(row)} |")
    lines.append("")

    lines.append("**Training time (ms)**")
    lines.append("")
    lines.append("| Variant | Vocab=500 | Vocab=1000 | Vocab=2000 | Vocab=4000 |")
    lines.append("|---------|-----------|------------|------------|------------|")
    for variant in ["GPT2", "GPT4"]:
        var_label = VARIANT_LABELS.get(variant, variant)
        row = []
        for vs in [500, 1000, 2000, 4000]:
            matches = [r for r in mojo_rows if r['variant'] == variant and r['vocab_size'] == vs]
            if matches:
                row.append(f"{fmt_ms(get(matches[0], 'train_ms'))}")
        if row:
            lines.append(f"| {var_label} | {' | '.join(row)} |")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = set(a for a in sys.argv[1:] if a.startswith("--"))

    if len(args) < 5:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    mojo = load_json_lines(args[0])
    py_tiktoken = load_json_lines(args[1])
    rs_tiktoken = load_json_lines(args[2])
    mbpe_py = load_json_lines(args[3])
    mojo_native = load_json_lines(args[4])
    show_hardware = "--no-hardware" not in flags

    # Corpus size from first available row
    n_bytes = 0
    for r in mojo + py_tiktoken + rs_tiktoken + mbpe_py + mojo_native:
        b = r.get('corpus_bytes', 0)
        if b:
            n_bytes = b
            break

    mb = n_bytes / 1_048_576 if n_bytes > 0 else 0
    if mb >= 1:
        corpus_label = f"{mb:.1f} MB"
    else:
        corpus_label = f"{n_bytes // 1024} KB"

    print(f"# Benchmark Results — {corpus_label} corpus")
    print()

    # 1. Hardware (skippable for multi-corpus runs)
    if show_hardware:
        print(make_hardware_table())
        print()

    # 2. Executive summary
    print("## Executive Summary")
    print()
    print(make_summary(mbpe_py, py_tiktoken, n_bytes))
    print()

    # 3. Combined comparison table
    print(make_comparison_table(mbpe_py, py_tiktoken, rs_tiktoken, mojo, mojo_native, n_bytes))
    print()

    # 4. Native Mojo pipeline (only if we have Mojo data)
    if mojo:
        print(make_mojo_table(mojo))
        print()

    # 5. Scaling
    if mojo:
        print(make_scaling_tables(mojo))
        print()


if __name__ == "__main__":
    main()
