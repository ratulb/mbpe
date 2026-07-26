# Benchmark Plan — Realistic Performance Characterization

## Current State

- Single corpus (`benchmarks/corpus.txt`, 1 MB, Alice in Wonderland)
- Mojo measures encode + decode best-of-20 throughput for one `BPE_PT` variant at a time
- Training timed once (no warmup, no min-of-runs)
- Python/Rust always use GPT-2 encoding regardless of `BPE_PT`
- No validation integrated into the benchmark run
- Pre-tokenization not measured independently

---

## 1. Input Corpora

Benchmarks must be credible across text types, not just prose.

| Corpus | Source | Target Size | Type | Purpose |
|--------|--------|-------------|------|---------|
| `corpus_small` | Alice in Wonderland | 1 MB | English prose | Baseline, fast iteration |
| `corpus_medium` | Wikipedia sample (~5 articles) | 10 MB | Mixed prose + markup | Mid-scale throughput |
| `corpus_code` | Source from CPython / Linux kernel | 10 MB | Code + comments | GPT-4 code-path stress |
| `corpus_mixed` | 50% prose + 30% code + 20% multilingual | 20 MB | Realistic heavy load | Final production proxy |
| `corpus_tiny` | First 10 KB of Alice | 10 KB | Pre-tokenization micro-benchmark | Split-only throughput |

**Download**: All auto-fetched by `run.sh` via curl from a known URL set.

---

## 2. Measurement Dimensions

### 2.1 Pre-tokenization (split-only throughput)
Isolate split speed from BPE merge overhead.

**Methodology:**
- Create a `PreTokenizer` instance, call `.split(text)` in a loop
- Report M chars/sec processed
- Test across all 3 PT variants: GPre, GPT2, GPT4
- Use all 5 corpora (shows how regex complexity scales with text type)

**Why separate?** Split speed is the bottleneck for GPT2/GPT4 (regex matching). GPre is ~100× faster, so including it as a baseline quantifies the cost of regex-based pre-tokenization.

### 2.2 Encode (full pipeline)
Pre-tokenize + BPE merge, end-to-end.

**Methodology:**
- Train tokenizer for each PT variant on the target corpus (vocab_size = training corpus type × 2, capped at 1000 for comparability)
- 3 warmup iterations, then 20 timed iterations
- Report best time (min), tokens/sec, and first-encode time (includes compile/JIT)
- Measure on `corpus_medium` and `corpus_mixed` (large enough for stable timings)

**Cross-variant comparison:**
- GPre (baseline) — fastest pre-tokenizer, no regex
- GPT2 — r50k_base regex, matches Python tiktoken GPT-2 reference
- GPT4 — cl100k_base regex, matches Python tiktoken cl100k_base reference

### 2.3 Decode (reconstruction throughput)
Token IDs → reconstructed string.

**Methodology:**
- 3 warmup, 20 timed iterations
- Report best time, tokens/sec
- Measure on same token sequences produced by 2.2

### 2.4 Training throughput
Merges learned per second.

**Methodology:**
- Train with vocab_size=500 on `corpus_medium`
- 5 fresh runs, report min time
- Metrics: merges/sec = len(merges) / train_time_s
- Vocab-steps/sec = (vocab_size - 256) / train_time_s

**Why 5 runs?** Training includes random-like loops (Dict iteration), min of 5 gives the noise floor.

### 2.5 .tiktoken save/load throughput
Serialization and deserialization speed at scale.

**Methodology:**
- Train on `corpus_medium` (vocab_size=1000)
- Save + load .tiktoken, 10 iterations each, report min
- Metrics: MB/sec (save), entries/sec (load)
- Also test with real o200k_base (200K entries) for scale

---

## 3. Comparison Targets

| Implementation | Variant | What it does |
|---------------|---------|-------------|
| **Mojo GPre** | `BPE_PT=0` | BPETokenizer with Ġ pre-tokenizer (baseline) |
| **Mojo GPT2** | `BPE_PT=1` | BPETokenizer with r50k_base regex |
| **Mojo GPT4** | `BPE_PT=2` | BPETokenizer with cl100k_base regex |
| **Python tiktoken** | `gpt2` | C-rust tiktoken, GPT-2 encoding |
| **Python tiktoken** | `cl100k_base` | C-rust tiktoken, GPT-4 encoding |
| **Rust tiktoken-rs** | `p50k_base` | Native Rust, GPT-2 encoding |

**Deferred** (infra not ready): Rust tiktoken-rs cl100k_base, Python tiktoken o200k_base.

---

## 4. Cross-Validation (integrated into benchmark run)

Every benchmark run must verify correctness, not just measure speed.

### 4.1 Token ID verification
- For each PT variant, after training, encode the first 100 KB of the corpus
- Compare token IDs against Python reference (same split + same merge table → same IDs)
- Fail the benchmark if any ID differs

### 4.2 Split count alignment
- Verify split counts match Python `regex` reference for all 3 PT variants
- Already done in `main.mojo` `test_split_counts` — re-verify during benchmark

### 4.3 Round-trip identity
- `decode(encode(text)) == text` for each benchmark input
- If mismatch, print the differing byte and abort

---

## 5. Scaling Analysis

### 5.1 Input length scaling
Measure encode throughput at input lengths: 1 KB, 10 KB, 100 KB, 1 MB, 10 MB.

**Expected behavior:**
- O(n) throughput (tok/s constant across sizes) for all implementations
- Deviations indicate super-linear hot paths (should be flagged as regressions)

### 5.2 Vocab size scaling
Train with vocab_size = 300, 500, 1000, 2000, 5000. Measure encode throughput.

**Expected:** Greedy merge loop scans up to N merges × word tokens per iteration. Throughput should decrease slightly with more merges.

### 5.3 Merge table sensitivity
Load pre-trained .tiktoken files with different merge counts:
- GPT-2 r50k_base (50,257 tokens, ~50K merges)
- cl100k_base (100,277 tokens, ~100K merges)
- o200k_base (200,019 tokens, ~200K merges)

**Expected:** PairCache guarantees O(1) lookup per pair, so throughput should be independent of vocab size.

---

## 6. Memory Profiling

Track peak RSS during:
- Training (corpus_medium, vocab_size=500)
- Encode (corpus_medium, single pass)
- Decode (corpus_medium, single pass)

Use `/proc/self/status VmPeak` (Linux) before and after each phase.

**Report:** Peak RSS per phase in MB. Flag leaks/regressions vs baseline.

---

## 7. Output Format

Unified table printed by `run.sh` after all variants complete:

```
╔══════════════════════════════════════════════════════════════════╗
║  simple_bpe — Performance Report                                ║
║  corpus_mixed  (20 MB)  |  vocab_size=500                      ║
╚══════════════════════════════════════════════════════════════════╝

── Pre-tokenization ──
  GPre     1,234.5 M char/s
  GPT2       245.6 M char/s
  GPT4       312.7 M char/s

── Encode ──
                                  best          tok/s       vs Python
  Mojo GPre                      12.3 ms      45.6 M/s        —
  Mojo GPT2                      45.6 ms      12.3 M/s     1.4×
  Python tiktoken (gpt2)         63.4 ms       8.9 M/s       1.0×
  Rust tiktoken-rs (p50k)        34.5 ms      16.2 M/s        —

── Decode ──
  Mojo GPre                       3.2 ms     178.9 M/s        —
  Mojo GPT2                       3.2 ms     178.9 M/s        —
  Python tiktoken (gpt2)          8.9 ms      64.3 M/s       1.0×
  Rust tiktoken-rs (p50k)        10.1 ms      56.7 M/s        —

── Training ──
  Mojo GPre                    234 ms         2.1 merges/ms
  Mojo GPT2                    312 ms         1.6 merges/ms
  Mojo GPT4                    298 ms         1.7 merges/ms

── Scaling (vocab size vs encode) ──
  Vocab    GPre          GPT2
   300     45.1 M/s      12.1 M/s
   500     44.8 M/s      11.9 M/s
  1000     44.5 M/s      11.7 M/s
  2000     43.9 M/s      11.4 M/s
  o200k    44.2 M/s      11.5 M/s

── Memory (peak RSS) ──
  Train     123 MB
  Encode     87 MB
  Decode     12 MB

── Validation ──
  Token IDs match Python reference:    ✔
  Split counts match reference:        ✔
  Round-trip identity:                 ✔
```

---

## 8. Implementation Order

### Milestone A — Infrastructure
1. Add split-only benchmark to `benchmark.mojo` (measure `.split()` throughput without encoding)
2. Extend `benchmark.mojo` to run all 3 PT variants in a single invocation (no more `-D` dispatch for the main comparison — run all 3 sequentially)
3. Add Python reference encode for GPT-4 (currently only does GPT-2)
4. Add scaling harness (easy to vary input length, vocab size)
5. Add memory measurement via `/proc/self/status`

### Milestone B — Multi-corpus support
6. Add `BENCHMARK_CORPORA` env var (colon-separated paths) or built-in URL list
7. Update `run.sh` to download and iterate over all corpora

### Milestone C — Validation integration
8. Add cross-validation step: train, save .tiktoken, load in Python, encode text, compare IDs
9. Fail benchmark on validation mismatch
10. Add round-trip identity check

### Milestone D — Reporting
11. Replace separate output capture (`mojo_result.txt`, `py_result.txt`, `rs_result.txt`) with structured JSON
12. Generate unified comparison table table
13. JSON export for CI/regression tracking

---

## 9. Prioritized Short-Term Wins (next session)

1. **Split-only micro-benchmark** — 1 file change, reveals regex pre-tokenization cost
2. **Run all 3 PT variants in one invocation** — replaces 3 separate `-D` runs
3. **Time training with min-of-5** — more reliable number
4. **Cross-validate token IDs** — add to `run.sh`, reuses `verify_encoding.py`
5. **Memory measurement** — simple `/proc/self/status` read before/after
6. **Scaling: vary vocab_size** — parameterize `train()` call in benchmark loop
