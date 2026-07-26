## Objective
- Comprehensive, multi-language benchmark suite across all 3 pre-tokenizer variants
- Scaling analysis: corpus size (10KB–5MB), vocab size (500–4000)
- Cross-language comparison: Mojo vs Python tiktoken vs Rust tiktoken-rs
- Plan for Python bindings (mirror tiktoken API)

## Important Details
- **All 78 tests pass**: 36 `main.mojo` + 9 `tests/test_tokenizer.mojo` + 33 `tests/exhaustive_tokenizer.mojo`
- **Source validation changes** committed: `train()` checks vocab_size<256, `decode()` bounds-checks token IDs, `_register_special_token()` rejects empty/duplicate, `decode()` uses `from_utf8_lossy`, `merge_cache` reset on retrain
- **Benchmark suite** outputs JSON lines (one per variant×vocab_size combo), collated by `collate.py` and `build_summary.py` into markdown tables
- **3 pre-tokenizer variants**: GPre (Ġ), GPT2 (r50k_base regex), GPT4 (cl100k_base regex)
- **6 corpus sizes**: 10KB, 100KB, 500KB, 1MB, 2MB, 5MB (generated from Alice in Wonderland)
- **4 vocab sizes per variant**: 500, 1000, 2000, 4000
- **3 implementations compared**: Mojo (ours), Python (`tiktoken`), Rust (`tiktoken-rs`)

## Key Results (5 MB corpus)

### Encode Throughput (M tok/s)
| Implementation | GPT2 encoding | vs Python |
|---|---|---|
| **Mojo** | **11.2** | **3.7×** |
| Python tiktoken | 3.0 | 1.0× |
| Rust tiktoken-rs | 3.1 | 1.0× |

### Decode Throughput (M tok/s)
| Implementation | GPT2 encoding | vs Python |
|---|---|---|
| **Mojo** | **107.7** | **3.7×** |
| Python tiktoken | 29.0 | 1.0× |
| Rust tiktoken-rs | 56.9 | 1.0× |

### Training Time Scaling (GPT2)
| Corpus | Vocab=500 | Vocab=4000 |
|---|---|---|
| 100KB | 91 ms | 850 ms |
| 1 MB | 935 ms | 8,115 ms |
| 5 MB | 4,824 ms | 43,703 ms |

## Work State

### Completed
- **All 78 tests** across Phase 1-3 (vocab integrity, training edge cases, encode/decode roundtrip, validation error-handling, property invariants, byte/unicode edge cases)
- **Source validation**: 4 changes to `tokenizer.mojo` (train check, decode bounds, special token validation, from_utf8_lossy)
- **`BENCHMARK_PLAN.md`** — comprehensive plan with 13 milestones
- **Multi-language benchmark suite** — Mojo (3 PTs × 4 vocab sizes) + Python tiktoken (gpt2 + cl100k) + Rust tiktoken-rs (p50k + cl100k), all producing parseable JSON
- **Scaling data** — 6 corpus sizes (10KB–5MB), 4 vocab sizes (500–4000), all 3 PT variants
- **Collation pipeline** — `collate.py` produces markdown tables, `build_summary.py` produces cross-size summary
- **`run.sh`** orchestrates all 3 implementations across all corpus sizes
- **Corpora generation** — `generate_corpora.py` produces 6 sizes from base corpus
- **`bm.mojo`** rewritten — no more `-D BPE_PT=N` dispatch, calls `run_all()` directly
- **`PYTHON_BINDINGS_PLAN.md`** — clean plan to expose BPETokenizer as `import simple_bpe` with tiktoken-compatible API
- **Git commit**: `3889961` on `incremental_stats` — 23 files, +3826/−690 lines

### Active
- Python bindings implementation (Phase 1-3 per plan)

### Blocked
- *(none)*

## Next Move
1. **Phase 1 Python bindings** — minimal v0: single `simple_bpe.so`, `BPETokenizer[GPreTokenizer]` exposed, `get_encoding("gpt2")` loads bundled .tiktoken, `train`/`encode`/`decode`/`save`/`load` work from Python
2. **Phase 2** — add GPT2 + GPT4 tokenizer types, module-level `train()` factory with `pretokenizer=` kwarg, `register_special_tokens`, `save_tiktoken`/`load_tiktoken`
3. **Phase 3** — tiktoken API parity: `decode_bytes`, `decode_single_token_bytes`, `token_byte_values`, `encode_single_token`, batch methods
4. **Benchmark: restore stable training iters** (currently 1 for speed; restore to 3 for final publication numbers)

## Relevant Files
- `/home/tenmoomnet/simple_bpe/tokenizer.mojo`: BPETokenizer[PT] — train, encode, decode, save/load, PairCache, special tokens
- `/home/tenmoomnet/simple_bpe/pretokenizer.mojo`: PreTokenizer trait, ByteMapping, 3 implementations
- `/home/tenmoomnet/simple_bpe/main.mojo`: 36 tests
- `/home/tenmoomnet/simple_bpe/tests/test_tokenizer.mojo`: 9 tests
- `/home/tenmoomnet/simple_bpe/tests/exhaustive_tokenizer.mojo`: 33 tests (Phase 1-3)
- `/home/tenmoomnet/simple_bpe/BENCHMARK_PLAN.md`: Benchmark plan with 13 milestones
- `/home/tenmoomnet/simple_bpe/PYTHON_BINDINGS_PLAN.md`: Python bindings plan (tiktoken-compatible API)
- `/home/tenmoomnet/simple_bpe/benchmarks/`: Full benchmark suite
  - `benchmark.mojo` — shared helpers, `run_all()`, `run_one[PT]()`
  - `bm.mojo` — entry point calling `run_all()`
  - `benchmark_tiktoken.py` — Python tiktoken (gpt2 + cl100k)
  - `benchmark_rust/` — Rust tiktoken-rs (p50k + cl100k)
  - `run.sh` — orchestrator across all corpora
  - `collate.py` — markdown table collation from JSON
  - `build_summary.py` — cross-size summary tables
  - `generate_corpora.py` — corpus size generation
- `/home/tenmoomnet/simple_bpe/SUMMARY.md`: This file — session summary and next moves
