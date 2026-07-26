## Objective
- Optimize BPETokenizer decode performance; refactor pre-tokenization into trait-backed dispatch; cross-verify all three pre-tokenizers against Python regex references; add incremental pair-stats training; document everything extensively.

## Important Details
- Decode optimized from 19.9 → **66.7 M tok/s** (first pass), then **109.3 M tok/s** with `memcpy` chain via `List.unsafe_ptr()` — 1.9× Rust tiktoken-rs (57.1 M). Encode 10.9 M tok/s (3.5× Rust).
- `PreTokenizer` trait + three implementations: `GPreTokenizer` (Ġ), `GPT2Pretokenizer` (r50k_base, 7 matchers), `GPT4Pretokenizer` (cl100k_base, 8 matchers).
- Single benchmark entry point `benchmarks/bm.mojo` with `-D BPE_PT=0/1/2` dispatch (env var `BPE_PT` in `run.sh`).
- Level A pre-tokenizer alignment verified — all three match Python `regex` references on curated strings AND full corpus counts.
- Level B end-to-end encode verification — `benchmarks/verify_encoding.py` produces identical token IDs on all 3 variants vs Python reference.
- GPT4Pretokenizer bugs fixed: `_best_match` → first-match-wins; added missing matchers with backtracking.
- GPT2Pretokenizer bug fixed: `_match_ws_not_before_nonws` backtracking.
- **`BPE_CORPUS` env var** added: all 3 benchmarks (Mojo, Python tiktoken, Rust tiktoken-rs) read it; URLs auto-downloaded to temp file by `run.sh` via curl, cleaned up on EXIT. Default fallback: `benchmarks/corpus.txt`.
- **`GAPS.md`**: comprehensive gap analysis vs bpe.mojo — incremental training stats, `.tiktoken` format, special tokens, byte_shuffle, pre-trained weights, Python-free save/load. All 6 gaps documented with background, mechanism details, bpe.mojo code snippets, and priority roadmap.
- **`INCREMENTAL_STATS.md`**: detailed 456-line design doc for incremental pair stats — flat `List[Int]` with SEP=-1 sentinel, packed `Dict[Int,Int]` pair-frequency map, 5-ops-per-occurrence incremental update, full worked example, performance analysis.
- **Incremental pair-stats implementation**: completed. Flat `List[Int]` with SEP=-1 replaces `Dict[String,List[Int]]` splits; packed `Dict[Int,Int]` pair-frequency map updated incrementally (5 ops per occurrence); single-pass merge scan replaces per-word rebuild. Old `_compute_pair_freqs()` and `_merge_pair()` helpers removed.
- Key design: decode uses precomputed `token_bytes` (flat `List[UInt8]` with `unsafe_ptr()` for raw pointer), `token_offsets`, `token_lengths` with Ġ→0x20 pre-substitution during train/load. No structural change needed for storage — `memcpy` on `List[UInt8]` via `unsafe_ptr()` eliminates ARC overhead.

## Work State
### Completed
- Decode profiling and optimization: sum pass, raw alloc, Ġ→0x20 pre-substitution, `memcpy` chain via `List.unsafe_ptr()` — 109.3 M tok/s (1.9× Rust).
- PreTokenizer trait + 3 implementations in `pretokenizer.mojo` with full docs (arch diagram, trait contract, per-matcher regex mapping, edge-case notes).
- GPT4Pretokenizer and GPT2Pretokenizer bugfixes (first-match-wins, matcher corrections, backtracking).
- Level A (4 alignment tests in `main.mojo`) and Level B (`benchmarks/verify_encoding.py`) verification — all pass.
- `BPE_CORPUS` env var with URL auto-download — supported by all 3 benchmarks (`benchmark.mojo`, `benchmark_tiktoken.py`, Rust `main.rs`), handled by `run.sh`.
- `GAPS.md`: 6-gap analysis vs bpe.mojo (incremental stats, .tiktoken, special tokens, byte_shuffle, pre-trained weights, Python-free save/load).
- `INCREMENTAL_STATS.md`: 456-line design doc with algorithm, worked example, performance analysis, bpe.mojo comparison.
- **Incremental pair-stats training**: flat `List[Int]` + SEP sentinel replaces `Dict[String,List[Int]]` splits; packed `Dict[Int,Int]` with single-pass incremental update replaces O(V×W) `_compute_pair_freqs` + `_merge_pair` helpers. All 10 tests pass; benchmark confirms identical token count (475,384) with training at 975 ms.

### Active
- **`.tiktoken` format support (gap #2)**: Design doc written (`TIKTOKEN_FORMAT.md`). Uses `from std.base64 import b64encode, b64decode` (stdlib, no copied module). 9-step plan: add `_bytes_key` / `_bpe` / `_recover_merges` → `save_tiktoken` → `load_tiktoken` → 3 tests → benchmark.

### Blocked
- *(none)*

## Next Move
1. **Implement `.tiktoken` format support** — see `TIKTOKEN_FORMAT.md` for the 10-step plan. Copy `encoding/base64.mojo` from bpe.mojo, then add `_bytes_key`, `_bpe`, `_recover_merges`, `save_tiktoken`, and `load_tiktoken` to `tokenizer.mojo`.
2. Train bpe.mojo-style `BasicTokenizer` on same corpus, compare merge rules, token counts, and convergence behavior against our incremental implementation.
3. Special-token support (gap #3 — `<|endoftext|>`, etc.).
4. `byte_shuffle` support (gap #4 — byte-level roundtrip for arbitrary Unicode).
5. Track metric: training throughput (vocab-steps/second) to quantify the incremental-stat speedup vs the old O(V×W) approach.

## Relevant Files
- `/home/tenmoomnet/simple_bpe/tokenizer.mojo`: 669 lines. `BPETokenizer[PT]` — `train()` (incremental pair stats), `encode()`, `decode()`, `PairCache`. Constants `ENCODE_SHIFT`, `ENCODE_MASK`, `SEP` at line 112.
- `/home/tenmoomnet/simple_bpe/pretokenizer.mojo`: PreTokenizer trait + 3 implementations (GPre, GPT2 r50k_base, GPT4 cl100k_base), `_is_ascii_ws_byte`, first-match-wins `_best_match`.
- `/home/tenmoomnet/simple_bpe/main.mojo`: 10 tests (6 original + 4 alignment).
- `/home/tenmoomnet/simple_bpe/GAPS.md`: Gap analysis vs bpe.mojo — 6 gaps documented with priority roadmap.
- `/home/tenmoomnet/simple_bpe/INCREMENTAL_STATS.md`: Detailed design doc for incremental pair stats — algorithm, worked example, performance analysis.
- `/home/tenmoomnet/simple_bpe/TIKTOKEN_FORMAT.md`: Design doc for .tiktoken format support — save/load, merge recovery algorithm, base64 via stdlib (`b64encode`/`b64decode`), 9-step implementation plan.
- `/home/tenmoomnet/simple_bpe/SUMMARY.md`: This file — session summary and next moves.
- `/home/tenmoomnet/simple_bpe/benchmarks/benchmark.mojo`: Shared helper `run[PT](label)`, reads `BPE_CORPUS` from env var.
- `/home/tenmoomnet/simple_bpe/benchmarks/run.sh`: Runner with dependency install, URL download, `BPE_CORPUS` propagation, `BPE_PT` mapping to `-D` flags.
- `/home/tenmoomnet/simple_bpe/benchmarks/benchmark_tiktoken.py`: Python tiktoken benchmark, reads `BPE_CORPUS` from env var.
- `/home/tenmoomnet/simple_bpe/benchmarks/benchmark_rust/src/main.rs`: Rust tiktoken-rs benchmark, reads `BPE_CORPUS` env var (higher priority than CLI arg).
- `/home/tenmoomnet/simple_bpe/PairCache.md`: Two-tier merge-lookup cache docs.
- `/home/tenmoomnet/simple_bpe/PRETOKENIZER.md`: Pre-tokenizer analysis and migration docs.
- `/home/tenmoomnet/simple_bpe/VERIFICATION.md`: Verification docs.
- `/home/tenmoomnet/simple_bpe/AGENTS.md`: Agent guide.
- `/home/tenmoomnet/../bpe.mojo/`: Sibling project. `BasicTokenizer` (pure byte-level BPE, incremental stats), `RegexTokenizer` (GPT-4 regex + special tokens + minbpe format), `GPT4Tokenizer` (tiktoken weight loading + byte_shuffle + recover_merges).
