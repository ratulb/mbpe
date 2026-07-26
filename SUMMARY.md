## Objective
- Implement comprehensive, realistic performance benchmarks across all 3 pre-tokenizer variants (GPre, GPT2, GPT4) against Python tiktoken and Rust tiktoken-rs, with integrated cross-validation.

## Important Details
- **Test file**: `/home/tenmoomnet/simple_bpe/tokenizer_tests.txt` — 510-line exhaustive test spec with 16 sections (~110 stubs: init, vocab, rank table, merges, training, incremental stats, encode, decode, roundtrip, special tokens, byte/unicode, .tiktoken I/O, cross-validation, property-based, performance, security/regression).
- **Already covered**: 45 tests (36 `main.mojo` + 9 `tests/test_tokenizer.mojo`). Detailed audit in `TEST_IMPLEMENTATION_PLAN.md`.
- **New tests planned**: ~20 Phase 1 (safe, no source changes), ~7 Phase 2 (needs validation/error-handling source changes), ~8 Phase 3 (complex edge cases, cross-validation).
- **Source validation gaps identified**: `train()` lacks vocab_size<256 check; `_register_special_token()` accepts empty/duplicate strings; `decode()` no bounds check; `load_tiktoken()` no malformed-line/duplicate/missing-file checks; no untrained-state guard for `save_tiktoken`/`encode`/`decode`.
- **45 existing tests pass** covering: pre-tokenizer alignment (all 3 variants, split counts), byte-level base vocab, encode/decode roundtrip, save/load JSON, determinism, Hugging Face corpus, .tiktoken format (structure, roundtrip, merge consistency, idempotency, JSON parity, all 3 PT variants, empty corpus, Unicode, zero merges), o200k_base interop (199,998 tokens), byte mapping (SEQUENTIAL/SHUFFLED roundtrip), special tokens (PT mappings, register, encode with/without, save/load, tiktoken skip, GPT2 auto-register, Wikipedia BPE example, single char, unicode roundtrip).

## Work State
### Completed
- All 3 phases of test implementation complete: 78 tests total (36 `main.mojo` + 9 `tests/test_tokenizer.mojo` + 33 `tests/exhaustive_tokenizer.mojo`). Coverage includes vocab integrity, training edge cases (emoji, mixed scripts, repeated chars, tie-breaking, oversized vocab), encode/decode (single byte, whitespace, large doc, stability, null byte, punctuation, unsplittable tokens), validation error-handling (vocab_size < 256, decode bounds, special token validation), property invariants (merge ranks, every-token-decodable), special token overlap, and byte/unicode edge cases.
- `BENCHMARK_PLAN.md` — comprehensive benchmark plan: 5 corpora, 6 measurement dimensions, 6 comparison targets, scaling analysis, integrated cross-validation, unified output table, 13 prioritized implementation milestones.

### Active
- Benchmark infrastructure upgrades per `BENCHMARK_PLAN.md`

### Blocked
- *(none)*

## Next Move
1. **Split-only micro-benchmark** — measure `.split()` throughput in isolation for all 3 PT variants
2. **Run all 3 PT variants in one invocation** — replace separate `-D` flag approach in `benchmark.mojo`
3. **Time training with min-of-5** — reliable training throughput number
4. **Cross-validate token IDs** — add to `run.sh`, reuse `verify_encoding.py`
5. **Memory measurement** — `/proc/self/status` peak RSS before/after each phase
6. **Scaling: vary vocab_size** — parameterize `train()` in benchmark loop
7. **Unified comparison table** — structured JSON output, printed table by `run.sh`

## Relevant Files
- `/home/tenmoomnet/simple_bpe/tokenizer.mojo`: ~979 lines. `BPETokenizer[PT]` — `train()`, `encode_ordinary()`, `encode()`, `decode()`, `save_tiktoken()`, `load_tiktoken()`, `PairCache`, `register_special_tokens()`.
- `/home/tenmoomnet/simple_bpe/pretokenizer.mojo`: ~1260 lines. `ByteMapping` enum, `PreTokenizer` trait, comptime SIMD LUTs, 3 implementations.
- `/home/tenmoomnet/simple_bpe/main.mojo`: ~810 lines, 36 tests.
- `/home/tenmoomnet/simple_bpe/tests/test_tokenizer.mojo`: 177 lines, 9 tests (Wikipedia BPE example, single char, unicode roundtrip).
- `/home/tenmoomnet/simple_bpe/tests/exhaustive_tokenizer.mojo`: ~490 lines, 33 tests (all phases).
- `/home/tenmoomnet/simple_bpe/BENCHMARK_PLAN.md`: Comprehensive benchmark plan with 13 milestones.
- `/home/tenmoomnet/simple_bpe/benchmarks/`: Benchmark suite — `benchmark.mojo` (shared helpers), `bm.mojo` (entry point), `benchmark_tiktoken.py` (Python reference), `benchmark_rust/` (Rust reference), `run.sh` (runner), `verify_encoding.py` (cross-validation), `reference_splits.py` (split alignment).
- `/home/tenmoomnet/simple_bpe/SUMMARY.md`: This file — session summary and next moves.
