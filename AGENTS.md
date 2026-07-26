# simple_bpe — Agent Guide

## Project

BPE tokenizer training in Mojo (from the Hugging Face course). Learns merge rules
on a corpus, then tokenizes new text using learned merges.

## Commands

All commands run inside `pixi shell` or prefixed with `pixi run`:
- **Run**: `mojo main.mojo`
- **Tests**: `mojo -I . tests/test_tokenizer.mojo` — 9 tests (byte-level, roundtrip, unicode, save/load, Wikipedia BPE example).
- **Exhaustive tests**: `mojo -I . tests/exhaustive_tokenizer.mojo` — Phase 1+2+3: 33 tests (vocab integrity, training edge cases, encode/decode, property invariants, validation error-handling, byte/unicode edge cases).
- **All tests**: run `pixi run mojo main.mojo && pixi run mojo -I . tests/test_tokenizer.mojo && pixi run mojo -I . tests/exhaustive_tokenizer.mojo` — 78 total.
- **Benchmark**: `pixi run benchmark` (GPreTokenizer), `pixi run benchmark-gpt2` (GPT2),
  `pixi run benchmark-gpt4` (GPT4).  Uses `-D BPE_PT=N` comptime flags
  (`from std.sys.defines import get_defined_int`) to select pre-tokenizer variant
  from a single entry point `benchmarks/bm.mojo`.
- **No linter, formatter, or typechecker** configured. No CI.

## Environment

- **Package manager**: Pixi (conda-based). Dependencies in `pixi.toml`.
- **Channels**: `https://conda.modular.com/max/`, `conda-forge`
- **Platform**: `linux-64`
- **Dependencies**: `mojo >=1.0.0b2,<2`, `python >=3.14.6,<3.15`
- **Setup**: `pixi install` (creates `.pixi/envs/default`)
- Ignored by git: `.pixi/*` (except `.pixi/config.toml`)

## Architecture

- `tokenizer.mojo` — `BPETokenizer` struct with `train()`/`encode()`/`decode()`,
  `save()`/`load()`, `save_tiktoken()`/`load_tiktoken()`. Module-level helpers
  `_compute_pair_freqs`, `_merge_pair`. Parameterized with `PT: PreTokenizer`
  — `BPETokenizer[GPreTokenizer]` default. Encode hot path uses comptime
  `PT.byte_map` branch for zero-overhead sequential mapping.
- `pretokenizer.mojo` — `ByteMapping` enum (SEQUENTIAL/SHUFFLED), `PreTokenizer` trait
  (split, byte_to_id, id_to_byte, 3 shared whitespace matchers), comptime SIMD LUTs
  for o200k byte permutation, three implementations:
  `GPreTokenizer` (Ġ), `GPT2Pretokenizer` (r50k_base), `GPT4Pretokenizer` (cl100k_base/o200k_base).
  `GPT4Pretokenizer[mapping]` parameter selects byte mapping at compile time.
- `benchmarks/bm.mojo` — single entry point dispatches via `get_defined_int["BPE_PT", 0]()`.
- `main.mojo` — tests via `TestSuite.discover_tests` (std.testing).
- `tests/test_tokenizer.mojo` — additional test suite (run with `-I .`).
- Pre-tokenizer approximates GPT-2's `Ġ` convention (space → `Ġ` prefix).
- Byte-level base vocab (no UNK — ID 0 is byte 0x00). All 256 bytes are base tokens.
- **Internals work with `Int` token IDs** — no string manipulation in hot loops:
  - `splits: Dict[String, List[Int]]` — each word is a list of token IDs
  - `merges: List[MergeRule]` — ordered `(first, second, merged)` (no dict-ordering footgun)
  - `_tokenize` returns `List[Int]` directly; `encode` is a passthrough for ordinary text,
    but scans for special tokens when registered
- **PairCache** — two-tier merge-lookup cache (flat `Int` array for IDs < 1000,
  `Dict` for larger IDs). Sentinel `-1` means no merge. Populated after each merge
  in `train()` and rebuilt in `load()`.
- **Special tokens**: Each `PreTokenizer` defines its encoding family's special
  tokens via `special_tokens()` static method (no default). Auto-registered on
  `load_tiktoken()`, skipped by `save_tiktoken()`. `encode()` scans input at
  byte level and preserves special tokens as single IDs. `register_special_tokens()`
  allows manual registration for custom tokenizers. 8 dedicated tests.
- Strings materialised only when needed: `vocab: List[String]` for decode,
  vocab-string concatenation for new merge entries, save/load.
- `encode(String) -> List[Int]`, `decode(Span[Int]) -> String` (replaces `Ġ`
  with space on decode).
- `Sized` trait enabled (`len(tok)` works). `Movable` for `^` transfer.
- Project history tracked in `CHANGE_LOG.md` (append-only, dated entries).

## Pretokenization

Our `PreTokenizer` uses the `Ġ` convention (space → `Ġ` prefix, decode
reverses). tiktoken instead uses a regex-based split that keeps whitespace
inline. A pure-Mojo UTF-8 state machine exists in `../bpe.mojo/bpe/pretokenizer.mojo`
but implements the cl100k_base pattern, not GPT-2's r50k_base.

**See `PRETOKENIZER.md` for the full analysis and migration plan.**

## Benchmarks

- `BPE_CORPUS` env var: file path or URL (auto-downloaded by `run.sh`, cleaned up on exit).
  Default: `benchmarks/corpus.txt`.
- Benchmark compares Mojo vs Python tiktoken vs Rust tiktoken-rs.
- `benchmarks/verify_encoding.py` cross-checks Mojo token IDs against Python reference.
- `benchmarks/reference_splits.py` generates reference split counts for alignment verification.

## Mojo conventions

- **Prefer `Span` over `List` in function parameters** — Span is a borrowed
  view and never copies the data.  `List[T]` converts to `Span[T]` implicitly
  via `__as_span`, so callers can pass either without extra work.

## Design docs

- `GAPS.md` — gap analysis vs `../bpe.mojo` (6 gaps: incremental stats ✅, .tiktoken ✅, special tokens ✅, byte_shuffle, pre-trained weights, Python-free save/load).
- `INCREMENTAL_STATS.md` — design doc for incremental pair stats (flat `List[Int]` with SEP sentinel).
- `TIKTOKEN_FORMAT.md` — `.tiktoken` format save/load plan (b64encode/b64decode from stdlib).
- `SUMMARY.md` — active work tracking and next moves.
