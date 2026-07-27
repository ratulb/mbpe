# Python test suite plan

## Goals

- Python tests matching Mojo test coverage (76 tests) + new tests for gaps
- `allowed_special`/`disallowed_special` params on `encode()`
- Same framework (pytest) works in GitHub Actions CI

## Framework

**pytest** + **pytest-cov** (added to pixi dev env).

## Architecture

### Python binding layer

The Mojo `.so` is built as `python-binding/_mbpe.so`. A Python package at `python-binding/mbpe/__init__.py` wraps it,
adding `allowed_special`/`disallowed_special` support in pure Python and delegating everything else via `__getattr__`.

### Test layout

```
tests/python/
├── conftest.py              # fixtures (4 tokenizer types, trained instances, paths)
├── test_core.py             # train, encode, decode, encode_ordinary, n_vocab, edge cases
├── test_serialization.py    # save_tiktoken, load_tiktoken roundtrip, structure
├── test_special_tokens.py   # register, encode with/without specials, auto-register
├── test_utility_methods.py  # name, decode_bytes, encode_single_token, token_byte_values, etc.
├── test_module_functions.py # get_encoding, mbpe.train, _train_impl
├── test_error_handling.py   # vocab<256, out-of-range IDs, empty specials, missing files
└── test_encode_params.py    # allowed_special, disallowed_special kwargs
```

### Coverage

| Test file | Mojo equivalent | Tests |
|-----------|-----------------|-------|
| test_core.py | main: 5-9, exhaustive: 44-63, test_tokenizer: 36-43 | ~12 |
| test_serialization.py | main: 10-22 | ~8 |
| test_special_tokens.py | main: 29-35, exhaustive: 69-71 | ~8 |
| test_utility_methods.py | GAPS A-F (no Mojo coverage) | ~6 |
| test_module_functions.py | GAPS G-I | ~5 |
| test_error_handling.py | exhaustive: 64-68 | ~5 |
| test_encode_params.py | NEW | ~5 |

### CI

`.github/workflows/python-tests.yml` uses `prefix-dev/setup-pixi@v0.8` to run pytest in the dev env.
