# simple_bpe — Agent Guide

BPE tokenizer training in Mojo (from the Hugging Face course).
`BPETokenizer[PT: PreTokenizer]` with 3 pre-tokenizer variants + Python bindings.

## Commands

All via `pixi run` or inside `pixi shell`:
- **All tests (78 total)**: `mojo main.mojo && mojo -I . tests/test_tokenizer.mojo && mojo -I . tests/exhaustive_tokenizer.mojo`
  - `main.mojo` = 36 tests (uses `TestSuite.discover_tests`)
  - `tests/test_tokenizer.mojo` = 9
  - `tests/exhaustive_tokenizer.mojo` = 33
- **Benchmark (full suite)**: `bash benchmarks/run.sh` — 4 implementations (Mojo, Python tiktoken, mbpe Python bindings, Rust tiktoken-rs) × 6 corpora
- **Quick benchmark** (1MB corpus, mbpe with 3 iters): `bash benchmarks/quick_bench.sh`
  - `pixi run benchmark` / `benchmark-gpt2` / `benchmark-gpt4` are aliases (all run `run.sh`)
  - `BPE_CORPUS` overrides corpus path (default: `benchmarks/corpus.txt`)
  - Outputs JSON lines to `benchmarks/results/`
  - Individual benchmarks: `benchmarks/bm.mojo` (Mojo), `benchmarks/benchmark_tiktoken.py` (tiktoken), `benchmarks/benchmark_mbpe.py` / `benchmark_mbpe_quick.py` (mbpe), `benchmarks/benchmark_rust/` (tiktoken-rs)
  - Collation: `python benchmarks/collate.py <mojo.json> <tiktoken.json> <tiktoken-rs.json> <mbpe.json>`
- **Benchmark setup**: `bash benchmarks/setup_bench_env.sh` — installs Rust (rustup) + Python venv at `/tmp/mbpe-bench-venv/` with tiktoken. Sourced by `run.sh` and `quick_bench.sh` automatically.
- **Direct benchmark**: `mojo -I . benchmarks/bm.mojo` (single corpus via `BPE_CORPUS`)
- **Reference splits**: `pixi run gen-refs`
- **Python bindings**: `mojo build mbpe.mojo --emit shared-lib -o mbpe.so`
  - Then: `PYTHONPATH=. python -c "import mbpe; tok = mbpe.GPreTokenizer()"`
  - API: 4 classes (`GPreTokenizer`, `GPT2Tokenizer`, `GPT4Tokenizer`, `GPT4oTokenizer`) each with `train`, `encode`, `encode_ordinary`, `decode`, `decode_bytes`, `decode_single_token_bytes`, `decode_with_offsets`, `n_vocab`, `name`, `save`, `load`, `save_tiktoken`, `load_tiktoken`, `register_special_tokens`, `encode_single_token`, `token_byte_values`
  - Module-level: `get_encoding("gpt2"|"cl100k"|"o200k")`, `train(texts, vocab_size)` (defaults to gpre). Use `_train_impl(texts, vocab_size, "gpt2"|"gpt4")` for specific pretokenizer.
  - `train()` on class instances accepts `vocab_size=` as keyword or positional argument
- **No linter / formatter / typechecker / CI**

## Environment

- `pixi install` (Mojo ≥1.0.0b2). Dev feature (`pixi shell --environment dev`) adds `tiktoken` for benchmarks.
- Channels: `https://conda.modular.com/max/`, `conda-forge`
- `.pixi/*` ignored by git (except `config.toml`)

## Architecture

- `tokenizer.mojo` — `BPETokenizer[PT: PreTokenizer = GPreTokenizer]`: train, encode, decode, save/load, save_tiktoken/load_tiktoken, `name()`, `decode_bytes()`, `encode_single_token()`, `token_byte_values()`
- `pretokenizer.mojo` — `PreTokenizer` trait, 3 impls: `GPreTokenizer` (Ġ), `GPT2Pretokenizer` (r50k_base), `GPT4Pretokenizer[ByteMapping]` (cl100k_base / o200k_base). Each has a `name()` static method.
- `mbpe.mojo` — Python bindings: 4 exported classes (`GPreTokenizer`, `GPT2Tokenizer`, `GPT4Tokenizer`, `GPT4oTokenizer`) + `get_encoding(name)` + `train(texts, vocab_size, pretokenizer='gpre')`
- `main.mojo` — 36 tests (entry point)
- Byte-level base vocab: all 256 bytes (0x00–0xFF), no UNK token
- Internals use `Int` token IDs; strings only on decode/save/load
- `PairCache` — two-tier merge lookup (flat array for IDs <1000, `Dict` otherwise)
- `encode(String) -> List[Int]`, `decode(Span[Int]) -> String`
- `ByteMapping`: `SEQUENTIAL` = identity (cl100k), `SHUFFLED` = permuted (o200k)

## Gotchas

- `test_load_o200k_base` hardcodes path `/home/tenmoomnet/bpe.mojo/data/o200k_base.tiktoken` — only on machine with sibling `bpe.mojo` checkout
- `test_split_counts` has expected split counts tied to `benchmarks/corpus.txt` content
- `ByteMapping` param must match `.tiktoken` file: `SEQUENTIAL` for cl100k, `SHUFFLED` for o200k

## Conventions

- Prefer `Span` over `List` in function parameters (zero-copy borrow; `List` converts implicitly)
- Tests in `tests/` need `-I .` for imports (not needed for `main.mojo`)
- `CHANGE_LOG.md` — append-only, dated entries

## Design docs (key references)

`GAPS.md` `PRETOKENIZER.md` `INCREMENTAL_STATS.md` `TIKTOKEN_FORMAT.md`
`BYTE_MAPPING_DESIGN.md` `PYTHON_BINDINGS_PLAN.md` `SUMMARY.md` `CHANGE_LOG.md`
