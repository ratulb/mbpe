# Session Backlog

Items to tackle in a future session, in rough priority order.

## High Value

1. **Training benchmark for mbpe_py** — Add train() timing to `benchmark_mbpe_quick.py` to measure FFI overhead during training (per-element corpus string conversion).

2. **Cross-platform wheels** — Build macOS + Windows wheels via cibuildwheel or manual CI matrix (currently linux x86_64 only; others fall back to the broken `any` wheel).

## Medium Value

3. **`encode_ordinary` + `decode_bytes` benchmark** — Add to benchmark scripts. `encode_ordinary` skips special tokens so it should be faster; `decode_bytes` returns `bytes` instead of `str`. Worth quantifying.

4. **Batch methods** — `encode_batch(texts)` / `decode_batch(ids_list)` — trivial loop wrappers, matches tiktoken API.

## Nice to Have

5. **Zero-copy corpus conversion in train()** — Apply `as_string_slice` + `PyList_GetItem` pattern to training loops (currently using `String(corpus_py[i])`).

## Done

- **`allowed_special`/`disallowed_special` encode params** — Implemented in `__init__.py` with full tiktoken-compatible logic.
- **CI pipeline** — `python-tests.yml` (push/PR) + `publish.yml` (release/manual) both working.
- **PyPI publishing** — `pyproject.toml`, `MANIFEST.in`, `scripts/publish.sh`, auditwheel repair with bundled Mojo runtime.
- **Benchmarks: all 4 impls in one table** — Mojo native pre-built, mbpe Python, tiktoken Python, tiktoken-rs. 3 iterations, o200k supported everywhere. collate.py streamlined.
