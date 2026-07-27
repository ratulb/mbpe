# Session Backlog

Items to tackle in a future session, in rough priority order.

## High Value

1. **Training benchmark for mbpe_py** — Add train() timing to `benchmark_mbpe_quick.py` to measure FFI overhead during training (per-element corpus string conversion).

2. **Python test suite** — `pytest` tests for all 4 tokenizer classes covering encode/decode/train/save/load roundtrips. Catches regressions on the Python side.

## Medium Value

3. **`encode_ordinary` + `decode_bytes` benchmark** — Add to benchmark scripts. `encode_ordinary` skips special tokens so it should be faster; `decode_bytes` returns `bytes` instead of `str`. Worth quantifying.

4. **Batch methods** — `encode_batch(texts)` / `decode_batch(ids_list)` — trivial loop wrappers, matches tiktoken API.

5. **`allowed_special`/`disallowed_special` encode params** — Currently auto-scans all registered specials. tiktoken lets users control which special tokens are allowed inline.

## Nice to Have

6. **Zero-copy corpus conversion in train()** — Apply `as_string_slice` + `PyList_GetItem` pattern to training loops (currently using `String(corpus_py[i])`).

7. **CI pipeline** — `.github/workflows/test.yml` running Mojo tests + Python tests on push/PR.
