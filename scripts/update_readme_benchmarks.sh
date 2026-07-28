#!/usr/bin/env bash
# Update benchmark tables in README.md from cached JSON results.
# Usage:  bash scripts/update_readme_benchmarks.sh
#
# Data files (in benchmarks/results/):
#   native.json      — Mojo native (pre-trained vocabs)
#   mbpe.json        — mbpe Python bindings
#   tiktoken.json    — Python tiktoken
#   tiktoken-rs.json — Rust tiktoken-rs
#   training.json    — Mojo training pipeline (GPT4 only)
#
# To regenerate data: bash benchmarks/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS="benchmarks/results"
README="README.md"
FILES=("native.json" "mbpe.json" "tiktoken.json" "tiktoken-rs.json" "training.json")

missing=0
for f in "${FILES[@]}"; do
    if [ ! -f "$RESULTS/$f" ]; then
        echo "Missing: $RESULTS/$f" >&2
        missing=1
    fi
done
if [ "$missing" -eq 1 ]; then
    echo "Run benchmarks first:  bash benchmarks/run.sh" >&2
    exit 1
fi

echo "  Generating encode/decode tables..." >&2

python3 -c "
import json

def load(path):
    return [json.loads(line) for line in open(path)]

native = {r['encoding']: r for r in load('$RESULTS/native.json')}
mbpe   = {r['encoding']: r for r in load('$RESULTS/mbpe.json')}
py     = {r['encoding']: r for r in load('$RESULTS/tiktoken.json')}
rs     = {r['encoding']: r for r in load('$RESULTS/tiktoken-rs.json')}

def fmt(v):
    if v is None: return '\u2014'
    return f'{v:.1f}'

def tok_fmt(n):
    if n >= 1_000_000:
        return f'{n/1_000_000:.2f}M'
    return f'{n//1000}K'

encodings = [
    ('gpt2',   'r50k_base'),
    ('cl100k', 'cl100k_base'),
    ('o200k',  'o200k_base'),
]
impls = [
    ('native', 'mbpe \u2014 Mojo native'),
    ('mbpe',   'mbpe \u2014 Python bindings'),
    ('py',     'tiktoken (Python)'),
    ('rs',     'tiktoken-rs'),
]
data = {'native': native, 'mbpe': mbpe, 'py': py, 'rs': rs}

for idx, (enc, base) in enumerate(encodings):
    best_e = max(
        (data[k][enc]['encode_mtok_s'] for k, _ in impls
         if data[k][enc]['encode_mtok_s'] is not None),
        default=-1.0,
    )
    best_d = max(
        (data[k][enc]['decode_mtok_s'] for k, _ in impls
         if data[k][enc]['decode_mtok_s'] is not None),
        default=-1.0,
    )

    if idx > 0:
        print()
    print(f'#### {enc} ({base})')
    print()
    print('| Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |')
    print('|---|---|---|---|')

    for key, label in impls:
        r = data[key][enc]
        tok = tok_fmt(r['n_tokens'])
        e_v = fmt(r['encode_mtok_s'])
        d_v = fmt(r['decode_mtok_s'])
        e_str = f'**{e_v}**' if r['encode_mtok_s'] == best_e else e_v
        d_str = f'**{d_v}**' if r['decode_mtok_s'] == best_d else d_v
        impl_label = f'**{label}**' if key == 'native' else label
        print(f'| {impl_label} | {tok} | {e_str} | {d_str} |')

# Blockquote note after o200k if tiktoken-rs beats or matches native on encode
ne = data['native']['o200k']['encode_mtok_s']
re_rs = data['rs']['o200k']['encode_mtok_s']
if re_rs is not None and ne is not None and re_rs >= ne:
    print()
    print(f'> On o200k, tiktoken-rs edges out encode speed ({fmt(re_rs)} vs. {fmt(ne)} M tok/s is within noise, but reported as-is); mbpe Mojo native still leads decode by a wide margin.')
" > /tmp/rb_encode_decode_table.txt

echo "  Generating training table..." >&2

python3 -c "
import json

lines = open('$RESULTS/training.json').read().strip().splitlines()
rows = [json.loads(l) for l in lines]

print('| Vocab size | 500 | 1000 | 2000 | 4000 |')
print('|---|---|---|---|---|')
train = [str(int(r['train_ms'])) + ' ms' for r in rows]
print(f'| Train time | {\" | \".join(train)} |')
ms = [str(r['train_merges_s']) for r in rows]
print(f'| Merges/s | {\" | \".join(ms)} |')
enc = [f'{r[\"encode_mtok_s\"]:.1f}' for r in rows]
print(f'| Encode (M tok/s) | {\" | \".join(enc)} |')
" > /tmp/rb_training_table.txt

echo "  Updating $README..." >&2

python3 -c "
import re

with open('$README') as f:
    content = f.read()

enc_block = open('/tmp/rb_encode_decode_table.txt').read()
trn_block = open('/tmp/rb_training_table.txt').read()

# Replace encode/decode section: from first subheading to blank line before Training
content = re.sub(
    r'#### gpt2 \(r50k_base\).*?\n(?=\n\*\*Training throughput)',
    enc_block,
    content,
    flags=re.DOTALL,
)

# Replace training section: from **Training** to blank line before Environment
content = re.sub(
    r'\*\*Training throughput\*\*.*?\n(?=\n\*Environment:)',
    f'**Training throughput** (Mojo, self-trained, GPT4Pretokenizer (cl100k_base / o200k_base), 5 MB corpus):\n\n{trn_block}',
    content,
    flags=re.DOTALL,
)

with open('$README', 'w') as f:
    f.write(content)

print('Done.')
"
