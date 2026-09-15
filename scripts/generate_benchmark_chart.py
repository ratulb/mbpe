#!/usr/bin/env python3
"""Generate docs/assets/benchmark_chart.svg from cached benchmark JSON.

Reads JSON-lines result files (one row per encoding) written by the
benchmark rig into benchmarks/results/:

  native.json      — Mojo native (pre-trained vocabs)
  mbpe.json        — mbpe Python bindings
  tiktoken.json    — Python tiktoken
  tiktoken-rs.json — Rust tiktoken-rs

and emits the grouped encode/decode bar chart embedded in README.md:

  <img src="docs/assets/benchmark_chart.svg" ...>

Layout mirrors the hand-made chart it replaces (960x560, left encode
panel + right decode panel, 3 encoding groups x 4 impls) so re-running
with unchanged numbers is a no-op diff. Stdlib only — no matplotlib.

Usage:
  python3 scripts/generate_benchmark_chart.py [--results DIR] [--output SVG]
"""
import argparse
import json
import os
import platform
import subprocess

ENCODINGS = ["gpt2", "cl100k", "o200k"]
GROUP_LABELS = {"gpt2": "gpt2 (r50k)", "cl100k": "cl100k", "o200k": "o200k"}
IMPLS = [
    ("native.json", "Mojo native", "#2563eb"),
    ("mbpe.json", "Py bindings", "#16a34a"),
    ("tiktoken.json", "tiktoken (Py)", "#f59e0b"),
    ("tiktoken-rs.json", "tiktoken-rs", "#ef4444"),
]

# Axis ceilings: first ladder rung >= max * HEADROOM. Ladders divide
# evenly by 4 (gridlines at max/4 steps). Current data lands on 20/220.
ENCODE_LADDER = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
DECODE_LADDER = [50, 100, 150, 200, 220, 250, 300, 400, 500]
HEADROOM = 1.08


def nice_max(peak, ladder):
    target = peak * HEADROOM
    for rung in ladder:
        if rung >= target:
            return rung
    return ladder[-1]


def fmt_val(v):
    s = f"{v:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return s


def fmt_grid(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return fmt_val(v)


def load_impl(results_dir, fname):
    path = os.path.join(results_dir, fname)
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by_enc = {r["encoding"]: r for r in rows}
    missing = [e for e in ENCODINGS if e not in by_enc]
    if missing:
        raise ValueError(f"{path} missing encodings: {missing}")
    return by_enc


def run(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def footer_env():
    def cpu_model():
        for line in run(["lscpu"]).split("\n"):
            if "Model name" in line:
                return line.split(":", 1)[-1].strip()
        return platform.processor() or "N/A"

    def mojo_ver():
        v = run(["pixi", "run", "mojo", "--version"])
        if v != "N/A":
            parts = v.split()
            return parts[1] if len(parts) >= 2 else v
        return "N/A"

    def tiktoken_ver():
        v = run(["pixi", "run", "-e", "dev", "python", "-c",
                 "import tiktoken; print(tiktoken.__version__)"])
        return v if v != "N/A" else "N/A"

    parts = [cpu_model()]
    tools = []
    mojo = mojo_ver()
    if mojo != "N/A":
        tools.append(f"Mojo {mojo}")
    tt = tiktoken_ver()
    if tt != "N/A":
        tools.append(f"tiktoken {tt}")
    if tools:
        parts.append(", ".join(tools))
    return "mbpe README benchmarks \u00b7 " + ", ".join(p for p in parts if p != "N/A")


def panel(out, cx, x0, x1, title, subtitle, amax, groups, key, legend=False):
    base, top = 470.0, 54.0
    plot_h = base - top
    gw = (x1 - x0) / 3.0
    bar_w, pitch, off = 27.3, 27.33, 15.0
    out.append(f'<text x="{cx}" y="28" text-anchor="middle" font-size="17"'
               f' font-weight="bold" font-family="sans-serif">{title}</text>')
    out.append(f'<text x="{cx}" y="46" text-anchor="middle" font-size="12"'
               f' fill="#555" font-family="sans-serif">{subtitle}</text>')
    for i in range(1, 5):
        gv = amax * i / 4.0
        y = round(base - plot_h * i / 4.0, 1)
        out.append(f'<line x1="{x0:g}" y1="{y:.1f}" x2="{x1:g}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x0 - 8:g}" y="{y + 4.0:.1f}" text-anchor="end"'
                   f' font-size="11" fill="#555" font-family="sans-serif">'
                   f'{fmt_grid(gv)}</text>')
    for g, enc in enumerate(ENCODINGS):
        gc = x0 + gw * g + gw / 2.0
        out.append(f'<text x="{gc}" y="492" text-anchor="middle" font-size="12"'
                   f' font-weight="bold" font-family="sans-serif">'
                   f'{GROUP_LABELS[enc]}</text>')
        for b, (_fname, label, color) in enumerate(IMPLS):
            v = groups[label][enc][key]
            h = plot_h * v / amax
            x = x0 + gw * g + off + pitch * b
            y = base - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}"'
                       f' width="{bar_w}" height="{h:.1f}"'
                       f' fill="{color}" rx="2"/>')
            out.append(f'<text x="{x + bar_w / 2.0:.1f}"'
                       f' y="{y - 4.0:.1f}" text-anchor="middle"'
                       f' font-size="10.5" font-weight="bold"'
                       f' font-family="sans-serif">{fmt_val(v)}</text>')
    out.append(f'<line x1="{x0:g}" y1="{base:g}" x2="{x1:g}" y2="{base:g}" stroke="#111"/>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results")
    ap.add_argument("--output", default="docs/assets/benchmark_chart.svg")
    args = ap.parse_args()

    for fname, _label, _color in IMPLS:
        if not os.path.isfile(os.path.join(args.results, fname)):
            raise SystemExit(
                f"Missing: {os.path.join(args.results, fname)} "
                f"(run benchmarks first: bash readme_bench_update.sh)")

    data = {label: load_impl(args.results, fname) for fname, label, _ in IMPLS}
    enc_peak = max(data[label][e]["encode_mtok_s"]
                   for _, label, _ in IMPLS for e in ENCODINGS)
    dec_peak = max(data[label][e]["decode_mtok_s"]
                   for _, label, _ in IMPLS for e in ENCODINGS)
    enc_max = nice_max(enc_peak, ENCODE_LADDER)
    dec_max = nice_max(dec_peak, DECODE_LADDER)

    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" role="img">',
           '<rect width="100%" height="100%" fill="white"/>']
    subtitle = "M tok/s, higher is better \u00b7 5MB Alice corpus, best-of-3"
    panel(out, 273.0, 64, 482, "Encode (M tok/s)", subtitle, enc_max, data, "encode_mtok_s")
    panel(out, 731.0, 522, 940, "Decode (M tok/s)", subtitle, dec_max, data, "decode_mtok_s")

    lx = [64, 284, 504, 724]
    for x, (_fname, label, color) in zip(lx, IMPLS):
        out.append(f'<rect x="{x}" y="510" width="14" height="14" fill="{color}"/>')
        out.append(f'<text x="{x + 20}" y="522" font-size="12"'
                   f' font-family="sans-serif">{label}</text>')
    out.append(f'<text x="940" y="550" text-anchor="end" font-size="10" fill="#888"'
               f' font-family="sans-serif">{footer_env()}</text>')
    out.append("</svg>")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {args.output} (encode_max={enc_max}, decode_max={dec_max})")


if __name__ == "__main__":
    main()
