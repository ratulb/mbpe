#!/usr/bin/env bash
# Regenerate the benchmark tables in README.md.
# Usage:  bash scripts/update_readme_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source benchmarks/setup_bench_env.sh >&2

BMDIR="benchmarks"
RESULTS_DIR="$BMDIR/results"
CORPUS="corpus_5MB.txt"
CORPUS_PATH="$BMDIR/$CORPUS"
mkdir -p "$RESULTS_DIR"

export BPE_CORPUS="$CORPUS_PATH"

echo "  [1/4] Mojo native (pre-trained vocabs)..." >&2
pixi run mojo -I . "$BMDIR/bm_pretrained.mojo" > /tmp/rb_native.json 2>/dev/null

echo "  [2/4] Python tiktoken..." >&2
"$VENV_PYTHON" "$BMDIR/benchmark_tiktoken.py" > /tmp/rb_tiktoken.json 2>/dev/null

echo "  [3/4] mbpe Python bindings..." >&2
pixi run python "$BMDIR/benchmark_mbpe_quick.py" > /tmp/rb_mbpe.json 2>/dev/null

echo "  [4/4] Rust tiktoken-rs..." >&2
(cd "$BMDIR/benchmark_rust" && cargo build --release 2>&1 | tail -1 >&2)
"$BMDIR/benchmark_rust/target/release/benchmark_rust" > /tmp/rb_rust.json 2>/dev/null

echo "  Building encode/decode table..." >&2

python3 -c "
import json

def load(path):
    return [json.loads(line) for line in open(path)]

native = {r['encoding']: r for r in load('/tmp/rb_native.json')}
mbpe   = {r['encoding']: r for r in load('/tmp/rb_mbpe.json')}
py     = {r['encoding']: r for r in load('/tmp/rb_tiktoken.json')}
rs     = {r['encoding']: r for r in load('/tmp/rb_rust.json')}

def fmt(v):
    if v is None: return '—'
    return f'{v:.1f}'

def tok_fmt(n):
    if n >= 1_000_000:
        return f'{n/1_000_000:.2f}M'
    return f'{n//1000}K'

encodings = ['gpt2', 'cl100k', 'o200k']

print('| Encoding | Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |')
print('|---|---|---|---|---|---|')

for enc in encodings:
    n = native[enc]
    tok_n = tok_fmt(n['n_tokens'])
    e_n = fmt(n['encode_mtok_s'])
    d_n = fmt(n['decode_mtok_s'])
    print(f'| **{enc}** | Mojo native | {tok_n} | **{e_n}** | **{d_n}** |')

    m = mbpe[enc]
    tok_m = tok_fmt(m['n_tokens'])
    e_m = fmt(m['encode_mtok_s'])
    d_m = fmt(m['decode_mtok_s'])
    print(f'| | mbpe (Python) | {tok_m} | {e_m} | {d_m} |')

    p = py[enc]
    tok_p = tok_fmt(p['n_tokens'])
    e_p = fmt(p['encode_mtok_s'])
    d_p = fmt(p['decode_mtok_s'])
    print(f'| | tiktoken (Python) | {tok_p} | {e_p} | {d_p} |')

    r = rs[enc]
    tok_r = tok_fmt(r['n_tokens'])
    e_r = fmt(r['encode_mtok_s'])
    d_r = fmt(r['decode_mtok_s'])
    print(f'| | tiktoken-rs | {tok_r} | {e_r} | {d_r} |')
" > /tmp/rb_encode_decode_table.txt

echo "  [5/4] Mojo training pipeline (5 MB)..." >&2
pixi run mojo -I . "$BMDIR/bm.mojo" > /tmp/rb_mojo_train.json 2>/dev/null

echo "  Building training table..." >&2

python3 -c "
import json

lines = open('/tmp/rb_mojo_train.json').read().strip().splitlines()
gpt4_rows = [json.loads(l) for l in lines if json.loads(l)['variant'] == 'GPT4']

print('| Vocab size | 500 | 1000 | 2000 | 4000 |')
print('|---|---|---|---|---|')
train = [str(int(r['train_ms'])) + ' ms' for r in gpt4_rows]
print(f'| Train time | {\" | \".join(train)} |')
ms = [str(r['train_merges_s']) for r in gpt4_rows]
print(f'| Merges/s | {\" | \".join(ms)} |')
enc = [f'{r[\"encode_mtok_s\"]:.1f}' for r in gpt4_rows]
print(f'| Encode (M tok/s) | {\" | \".join(enc)} |')
" > /tmp/rb_training_table.txt

README="README.md"
echo "  Updating $README..." >&2

python3 -c "
import re

with open('$README') as f:
    content = f.read()

enc_block = open('/tmp/rb_encode_decode_table.txt').read().strip()
trn_block = open('/tmp/rb_training_table.txt').read().strip()

# Replace encode/decode table (between markers)
content = re.sub(
    r'<!-- BENCH_ENCODE_DECODE_START -->.*?<!-- BENCH_ENCODE_DECODE_END -->',
    f'<!-- BENCH_ENCODE_DECODE_START -->\\n{enc_block}\\n<!-- BENCH_ENCODE_DECODE_END -->',
    content,
    flags=re.DOTALL,
)

# Replace training table (between markers)
content = re.sub(
    r'<!-- BENCH_TRAINING_START -->.*?<!-- BENCH_TRAINING_END -->',
    f'<!-- BENCH_TRAINING_START -->\\n**Training throughput** (Mojo, self-trained, GPT4Pretokenizer (cl100k_base / o200k_base), 5 MB corpus):\\n\\n{trn_block}\\n<!-- BENCH_TRAINING_END -->',
    content,
    flags=re.DOTALL,
)

with open('$README', 'w') as f:
    f.write(content)

print('Done.')
"
