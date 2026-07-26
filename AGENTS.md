# simple_bpe — Agent Guide

## Project

BPE tokenizer training in Mojo (from the Hugging Face course). Learns merge rules
on a corpus, then tokenizes new text using learned merges.

## Commands

All commands run inside `pixi shell` or prefixed with `pixi run`:
- **Run**: `mojo main.mojo`
- **Benchmark**: `pixi run benchmark` (GPreTokenizer), `pixi run benchmark-gpt2` (GPT2),
  `pixi run benchmark-gpt4` (GPT4).  Uses `-D BPE_PT=N` comptime flags
  (`from std.sys.defines import get_defined_int`) to select pre-tokenizer variant
  from a single entry point `benchmarks/bm.mojo`.
- **No test framework, linter, formatter, or typechecker** configured. No CI.

## Environment

- **Package manager**: Pixi (conda-based). Dependencies in `pixi.toml`.
- **Channels**: `https://conda.modular.com/max/`, `conda-forge`
- **Platform**: `linux-64`
- **Dependencies**: `mojo >=1.0.0b2,<2`, `python >=3.14.6,<3.15`
- **Setup**: `pixi install` (creates `.pixi/envs/default`)
- Ignored by git: `.pixi/*` (except `.pixi/config.toml`)

## Architecture

- `tokenizer.mojo` — `BPETokenizer` struct with `train()`/`encode()`/`decode()`,
  `save()`/`load()`. Module-level helpers `_compute_pair_freqs`, `_merge_pair`.
  Parameterized with `PT: PreTokenizer` — `BPETokenizer[GPreTokenizer]` default.
- `pretokenizer.mojo` — `PreTokenizer` trait + three implementations:
  `GPreTokenizer` (Ġ), `GPT2Pretokenizer` (r50k_base), `GPT4Pretokenizer` (cl100k_base).
  Pre-tokenizer type selected at compile time via `-D BPE_PT=N`.
- `benchmarks/bm.mojo` — single entry point dispatches via `get_defined_int["BPE_PT", 0]()`.
- `main.mojo` — tests via `TestSuite.discover_tests` (std.testing).
- Pre-tokenizer approximates GPT-2's `Ġ` convention (space → `Ġ` prefix).
- Character-level base vocab with `<UNK>` at ID 0 (unknown codepoints → 0).
- **Internals work with `Int` token IDs** — no string manipulation in hot loops:
  - `splits: Dict[String, List[Int]]` — each word is a list of token IDs
  - `merges: List[Tuple[Int, Int, Int]]` — ordered `(a_id, b_id, merged_id)` (no dict-ordering footgun)
  - `char_to_id: Dict[Int, Int]` — codepoint → base token ID
  - `_tokenize` returns `List[Int]` directly; `encode` is a passthrough
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

## Mojo conventions

- **Prefer `Span` over `List` in function parameters** — Span is a borrowed
  view and never copies the data.  `List[T]` converts to `Span[T]` implicitly
  via `__as_span`, so callers can pass either without extra work.
