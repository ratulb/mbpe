# Verification Strategy — Pre-tokenizer & BPE Cross-Reference

## What we are trying to prove

Our `BPETokenizer[PT]` has three independent layers:

| Layer | Risk | How to verify |
|---|---|---|
| **Pre-tokenizer** (regex/state machine) | Wrong split boundaries → bad BPE training + wrong encodings | Compare split output against Python `regex` module using same patterns (Level A) |
| **BPE merge learning** (`train()`) | Wrong pair-frequency counting, wrong tie-breaking | Covered indirectly by roundtrip + deterministic invariant; tie-breaking differences produce valid but different tokenizers |
| **BPE merge application** (`_tokenize()` / `encode()`) | Given correct splits and merges, produces wrong token IDs | Train in Mojo → save merges → Python applies same merges → compare token IDs (Level B) |

## Reference source considerations

### Pre-tokenizer splits — ground truth

tiktoken does regex split + BPE in one fused C/Rust call. We use Python `regex` module
with the same tiktoken patterns:

| Variant | Pattern source |
|---|---|
| GPreTokenizer | Replicate Mojo logic (space→Ġ, .→separate, split) |
| GPT2Pretokenizer | `(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s` |
| GPT4Pretokenizer | `(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s` |

**Loophole**: Python `regex` module ≠ Rust `regex` crate ≠ Mojo UTF-8 state machine.
Edge-case Unicode (zero-width joiners, regional indicators, obscure scripts) may differ.
Cross-check a random subset against tiktoken byte output if needed.

### BPE merge application — ground truth

The encode algorithm is *deterministic* given the same merge rules:
```
for each word (list of byte IDs 0–255):
    while adjacent pair (a, b) exists in merge cache:
        find pair with lowest merged_id (earliest merge)
        merge it
```

At each step, the lowest-merged_id pair is uniquely determined by the current token
sequence. Rank is total order → no tie-breaking during encoding.

**Proof**: train in Mojo → save merges as JSON → Python loads same merges →
apply same algorithm → token IDs must match bit-for-bit.

## Verification matrix

### Level A — Pre-tokenizer split alignment

| # | Scope | Input | Check |
|---|---|---|---|
| A1 | All 3 variants | Curated string (contractions, numbers, punct, ws) | Split output matches Python reference token-by-token |
| A2 | GPT2/GPT4 | Full corpus | Split count matches Python reference |
| A3 | GPT2/GPT4 | Full corpus | Random subset of splits match exactly |

### Level B — End-to-end encoding with shared merges

| # | Scope | Input | Check |
|---|---|---|---|
| B1 | All 3 variants | Train + save merges in Mojo | Python loads merges, encodes test string → same token IDs |
| B2 | All 3 variants | Full corpus | encode + decode on corpus → exact roundtrip |

### Level C — Continuous verification

| # | What |
|---|---|
| C1 | Wire Level A into `main.mojo` as regular test functions |
| C2 | Wire Level B into `main.mojo` |

## What passing does NOT prove

- **Training tie-breaking**: Training may break frequency ties differently than a
  reference BPE trainer. The resulting tokenizer is still valid (roundtrip holds).
  Different tie-breaking → different merges → different token IDs for the same input.
  This is expected and harmless — only the encode *algorithm* is verified, not the
  *training outcome*.
- **Full Unicode coverage**: The Wikipedia corpus is English-heavy. CJK, emoji, RTL
  text, and zero-width joiners are not exercised by corpus tests. Unit tests with
  specific Unicode strings (planned separately) cover this.
- **Performance**: Throughput measurement is separate from correctness.
  Optimizations (pre-computed offsets, byte-level storage) don't affect correctness
  beyond what `decode(encode(t)) == t` catches.
- **Python `regex` vs Rust `regex`**: The Python `regex` module may handle certain
  Unicode edge cases differently from the Rust `regex` crate that tiktoken uses.
  If mismatches appear, the second step is to cross-check against tiktoken's own
  `_encode_via_mergeable` byte output.

## Execution check-list (all ✅)

- [x] Save VERIFICATION.md
- [x] Write `benchmarks/reference_splits.py` — generates reference splits using Python `regex`
- [x] Write Mojo tests for Level A (pre-tokenizer alignment) — `test_gpre_splits`, `test_gpt2_splits`, `test_gpt4_splits`, `test_split_counts`
- [x] Run Level A, fix any mismatches
- [x] Write `benchmarks/verify_encoding.py` — loads Mojo merges, applies encode in Python
- [x] Write Mojo tests for Level B (end-to-end encoding) — `test_tiktoken_roundtrip`, `test_tiktoken_vs_json_parity`
- [x] Run Level B, fix any mismatches
- [x] Wire both into `main.mojo` (Level C)
- [x] Full test suite pass: 78 tests (36 main.mojo + 9 test_tokenizer + 33 exhaustive_tokenizer)
