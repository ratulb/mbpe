# Contributing to mbpe

Thanks for your interest in mbpe. This page covers how to set up a dev
environment, run the test suite, and submit changes. It is kept deliberately
short; the codebase is small and the test suite is the source of truth.

## Setup

Linux x86_64 is required. The toolchain is managed by
[pixi](https://pixi.sh) (Mojo compiler and a dev Python with `pytest` +
`tiktoken`):

```bash
git clone https://github.com/ratulb/mbpe
cd mbpe
pixi install
```

## Building the Python binding

The Python API is a Mojo-compiled shared library that is gitignored; build it
before running any Python code:

```bash
pixi run mojo build python-binding/mbpe.mojo -I . \
  --emit shared-lib -o python-binding/mbpe/_mbpe.so
```

The `-I .` flag is required so the `bpe/` module resolves.

## Running tests

One entrypoint runs everything that CI runs:

```bash
bash scripts/run_tests.sh
```

It rebuilds the shared library, then runs:

- the Python suite (parametrized over all three encodings, gpt2/cl100k/o200k),
  including the byte-for-byte parity checks against OpenAI's `tiktoken`;
- the Mojo suites: `main.mojo`, `tests/test_tokenizer.mojo`,
  `tests/exhaustive_tokenizer.mojo`.

The parity suite is the most important: `tests/python/test_parity_*.py`
cross-check mbpe against `tiktoken` for every encoding. If a change moves a
single token ID, these fail.

Individual commands:

```bash
pixi run mojo main.mojo
pixi run mojo -I . tests/test_tokenizer.mojo
pixi run mojo -I . tests/exhaustive_tokenizer.mojo
pixi run --environment dev python -m pytest tests/python/ -v
```

## Benchmarks

`pixi run benchmark` runs the full comparison suite (Mojo native, mbpe
bindings, `tiktoken`, `tiktoken-rs`) across several corpora. It regenerates
missing corpora and Rust toolchain automatically, and can take a while.

## Reporting bugs / requesting features

Open an issue. A minimal reproduction is ideal — a few lines of Python using
`mbpe.get_encoding("...")` plus the expected and actual token IDs.

## Submitting changes

1. Create a branch off `main`.
2. Make your change; follow the existing style (no formatting tooling is
   imposed — match the surrounding code).
3. Run `bash scripts/run_tests.sh` and make sure it is green.
4. Open a pull request with a short description of the change and, for
   behaviour changes, the tests that cover it.

## Code of conduct

Be kind and constructive. This is a small project run by one maintainer;
patience is appreciated.
