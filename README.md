# mbpe

[![Tests](https://github.com/ratulb/mbpe/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ratulb/mbpe/actions/workflows/python-tests.yml)
[![CodeQL](https://github.com/ratulb/mbpe/actions/workflows/codeql.yml/badge.svg)](https://github.com/ratulb/mbpe/actions/workflows/codeql.yml)
[![PyPI version](https://img.shields.io/pypi/v/mbpe)](https://pypi.org/project/mbpe/)
<!--[![PyPI downloads](https://img.shields.io/pypi/dm/mbpe)](https://pypi.org/project/mbpe/)-->

A **high-performance**, trainable, tiktoken-compatible BPE tokenizer written in **Mojo**.
The compile-time `PreTokenizer` trait enables GPT-2, GPT-4, GPT-4o and custom tokenization pipelines without changing the core tokenizer.

```python
import mbpe

tokenizer = mbpe.get_encoding("gpt2")
tokens = tokenizer.encode("hello world")
print(tokens)                          # [31373, 995]
print(tokenizer.decode(tokens))        # "hello world"
```

---

## Why mbpe?

- **Drop-in for [`tiktoken`](https://github.com/openai/tiktoken)** — same `get_encoding()`, same `encode()`/`allowed_special`/`disallowed_special`, same `.tiktoken` file format. Point existing code at `mbpe` and it works.
- **Fast** — Mojo native beats [`tiktoken-rs`](https://github.com/zurawiki/tiktoken-rs) (Rust) on both encode and decode across all three encodings. See [Benchmarks](#benchmarks)
- **Fast Python bindings** - Substantially outperform Python `tiktoken`, while still remaining competitive with `tiktoken-rs`.
- **Train your own** — `tokenizer.train(["hello world"], vocab_size=300)`, then save directly to `.tiktoken` format.
- **Extensible by design** — `PreTokenizer` is a Mojo trait, not a hardcoded implementation. Ships with r50k_base, cl100k_base, and o200k_base; write your own to match it.
- **Byte-level, lossless** — all 256 bytes are base vocabulary. No UNK token. Any valid UTF-8 input round-trips exactly.

---

## Benchmarks

5 MB corpus (Alice in Wonderland), best-of-3 encode + decode, pre-trained 50K+ vocabularies.

> **Across all three OpenAI encodings, native Mojo is consistently the fastest implementation for both encoding and decoding, while the Python bindings substantially outperform Python tiktoken**.

#### gpt2 (r50k_base)

| Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|---|
| **mbpe — Mojo native** | 1.54M | **14.4** | **169.4** |
| mbpe — Python bindings | 1.54M | 12.2 | 95.2 |
| tiktoken (Python) | 1.54M | 5.1 | 41.4 |
| tiktoken-rs | 1.53M | 4.7 | 70.6 |

#### cl100k (cl100k_base)

| Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|---|
| **mbpe — Mojo native** | 1.28M | **11.9** | **166.1** |
| mbpe — Python bindings | 1.28M | 9.9 | 85.3 |
| tiktoken (Python) | 1.28M | 4.6 | 39.9 |
| tiktoken-rs | 1.28M | 4.5 | 75.1 |

#### o200k (o200k_base)

| Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|---|
| **mbpe — Mojo native** | 1.28M | **9.6** | **164.8** |
| mbpe — Python bindings | 1.28M | 8.2 | 84.9 |
| tiktoken (Python) | 1.28M | 7.1 | 46.4 |
| tiktoken-rs | 1.28M | 8.1 | 78.1 |

**Training throughput** (Mojo, self-trained, GPT4Pretokenizer (cl100k_base / o200k_base), 5 MB corpus):

| Vocab size | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|
| Train time | 39 ms | 43 ms | 50 ms | 67 ms |
| Merges/s | 6104 | 17184 | 34528 | 55201 |
| Encode (M tok/s) | 25.8 | 19.5 | 15.8 | 14.0 |

*Environment: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, 4 cores, 7.8Gi RAM, Debian GNU/Linux 13 (trixie). Mojo 1.0.0b2, Python 3.14.6, Rust 1.97.1, tiktoken 0.13.0.*

---

> ⭐ — Helps others discover it!

---

## Installation

```bash
pip install mbpe
```

Linux x86_64 only (Mojo runtime bundled). Requires Python ≥ 3.9.

**Mojo (conda):**

```bash
pixi add mbpe --channel https://repo.prefix.dev/modular-community
```

Requires `mojo-compiler >=1.0.0b2`. The `.tiktoken` data files are auto-discovered when the environment is activated (via `MBPE_DATA_DIR`). Set this env var manually if using outside conda.

---

## Usage

### Load a pre-built encoding

```python
import mbpe

# GPT-2 (r50k_base, ~50K vocab)
tokenizer = mbpe.get_encoding("gpt2")
print(tokenizer.n_vocab)                          # 50257
print(tokenizer.encode("hello world"))            # [31373, 995]

# GPT-4 (cl100k_base, ~100K vocab)
tokenizer = mbpe.get_encoding("cl100k")
print(tokenizer.n_vocab)                          # 100277

# GPT-4o (o200k_base, ~200K vocab)
tokenizer = mbpe.get_encoding("o200k")
print(tokenizer.n_vocab)                          # 200019
```

### Train from scratch

```python
tokenizer = mbpe.GPT2Tokenizer()
corpus = ["the cat sat on the mat", "the dog sat on the log"]
tokenizer.train(corpus, vocab_size=300)
print(tokenizer.encode("the cat sat"))            # [258, 269, 263]
tokenizer.save_tiktoken("my_tokenizer.tiktoken")

# Or train with a GPT-4-style `PreTokenizer`
tokenizer = mbpe.GPT4Tokenizer()
tokenizer.train(corpus, vocab_size=300)
```

### Encode with special tokens

```python
tokenizer = mbpe.get_encoding("gpt2")

# Encode handles special tokens (e.g. <|endoftext|>)
tokenizer.encode("<|endoftext|>hello world")      # [50256, 31373, 995]

# encode_ordinary ignores them
tokenizer.encode_ordinary("<|endoftext|>hello")   # [27, 91, 437, 1659, 5239, 91, 29, 31373]

# Control which specials are allowed
tokenizer.encode("<|endoftext|>hello world", allowed_special={"<|endoftext|>"})
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")   # default
```

### Decode

```python
tokenizer = mbpe.get_encoding("gpt2")

tokenizer.decode([31373, 995])                    # "hello world"
tokenizer.decode_bytes([31373, 995])              # b"hello world"
tokenizer.decode_with_offsets([31373, 995])       # ("hello world", [(0, 5), (5, 11)])
tokenizer.decode_single_token_bytes(31373)        # b"hello"
```

### Save and load

```python
tokenizer = mbpe.get_encoding("gpt2")
tokenizer.save_tiktoken("backup.tiktoken")

tokenizer2 = mbpe.GPT2Tokenizer()
tokenizer2.load_tiktoken("backup.tiktoken")
assert tokenizer2.n_vocab == tokenizer.n_vocab
```

### Register custom special tokens

```python
tokenizer = mbpe.get_encoding("gpt2")
tokenizer.register_special_tokens({"<|im_start|>": 50257, "<|im_end|>": 50258})
print(tokenizer.encode("<|im_start|> hello"))     # [50257, 23748]
```

---

### Mojo API:

#### Load an encoding

BPETokenizer[PT] / Tokenizers.get[PT] → returns comptime parameterized tokenizer. Loads from `.tiktoken` files located via `MBPE_DATA_DIR` env var, falling back to `./data/`

```mojo
from bpe.tokenizer import Tokenizers


def main() raises:
    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    var ids = gpt2.encode("hello world")
    print(gpt2.decode(ids))
```
#### Train:

```mojo
from bpe.tokenizer import BPETokenizer
from bpe.pretokenizer import GPT2Pretokenizer


def main() raises:
    var tok = BPETokenizer[GPT2Pretokenizer]()
    var corpus = List[String]()
    corpus.append("hello world")
    tok.train(corpus, 300)
```

---

### Try it yourself

[`benchmarks/compare_mbpe_tiktoken.py`](benchmarks/compare_mbpe_tiktoken.py) — self-contained script that downloads a corpus from Project Gutenberg and compares mbpe vs tiktoken encode/decode throughput. Run it anywhere:

```bash
pip install mbpe tiktoken
python benchmarks/compare_mbpe_tiktoken.py
```

Paste the body into a Colab or Kaggle cell (prepend `!pip install mbpe tiktoken`) for the same comparison in a notebook.

---
## Architecture

At the core of mbpe is `BPETokenizer[PT]`, where `PT` is any implementation of the compile-time `PreTokenizer` trait.

```
                                         Text
                                          │
                                          ▼
                                  PreTokenizer(trait)
                                          │
                                          ▼
                                   Symbol Sequence
                                          │
                                          ▼
                                   BPETokenizer[PT]
                                 ┌────────┴─────────┐
                                 ▼                  ▼
                             Training          Encode/Decode

```
> A `PreTokenizer` converts input text into a sequence of symbols (e.g. regex chunks, Ġ-prefixed words, or future custom segmentations) before BPE merging begins.


### comptime aliases:

- `Tokenizers.get[Tokenizers.gpt2]()`  → `BPETokenizer[GPT2Pretokenizer]` → r50k_base
- `Tokenizers.get[Tokenizers.cl100k]()` → `BPETokenizer[GPT4Pretokenizer[SEQUENTIAL]]` → cl100k_base
- `Tokenizers.get[Tokenizers.o200k]()`  → `BPETokenizer[GPT4Pretokenizer[SHUFFLED]]` → o200k_base


---

## Implementation highlights

- Compile-time `PreTokenizer` trait dispatch
- Two-tier `MergeLookup` cache (flat array + Dict)
- Incremental pair statistics for training - (O(N) vs O(V×W))
- Flat memcpy-chain decoder
- Byte-level, lossless vocabulary

---

## Development

```bash
git clone https://github.com/ratulb/mbpe
cd mbpe

# Install dependencies
pixi install

# Build Python shared library
pixi run mojo build python-binding/mbpe.mojo -I . \
  --emit shared-lib -o python-binding/mbpe/_mbpe.so

# Run Mojo tests (77 total)
# pixi run mojo main.mojo            # 33 tests
# pixi run mojo -I . tests/test_tokenizer.mojo           # 10 tests
# pixi run mojo -I . tests/exhaustive_tokenizer.mojo     # 34 tests

# Run Python tests
# pixi run --environment dev python -m pytest tests/python/ -v
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started, run the suite, and open a pull request.

---

## API

| Python class | Mojo backend | Pre-tokenizer | `name()` |
|---|---|---|---|
| `GPT2Tokenizer` | `BPETokenizer[GPT2Pretokenizer]` | r50k_base regex (7 patterns) | `"gpt2"` |
| `GPT4Tokenizer` | `BPETokenizer[GPT4Pretokenizer[SEQUENTIAL]]` | cl100k_base regex (8 patterns) | `"cl100k"` |
| `GPT4oTokenizer` | `BPETokenizer[GPT4Pretokenizer[SHUFFLED]]` | o200k_base regex (8 patterns, shuffled byte mapping) | `"o200k"` |

All three classes share the same method interface:

| Method | Description |
|---|---|
| `train(texts, vocab_size)` | Train BPE from scratch |
| `encode(text, **kwargs)` | Encode with special token handling |
| `encode_ordinary(text)` | Encode ignoring special tokens |
| `decode(ids)` | Decode token IDs to string |
| `decode_bytes(ids)` | Decode to raw `bytes` |
| `decode_single_token_bytes(id)` | Raw bytes for one token |
| `decode_with_offsets(ids)` | `(text, [(start, end), ...])` — decoded text with byte offsets per token |
| `encode_single_token(text)` | Look up a token string's ID |
| `token_byte_values()` | Raw bytes for all token IDs |
| `name()` | Pre-tokenizer name |
| `n_vocab` (property) | Vocabulary size |
| `save_tiktoken(path)` | Save in .tiktoken format |
| `load_tiktoken(path)` | Load from .tiktoken file |
| `register_special_tokens(dict)` | Register special tokens |

Module-level functions:

| Function | Description |
|---|---|
| `get_encoding(name)` | Load pre-trained encoding (gpt2, cl100k, o200k) |
| `train(texts, vocab_size)` | Train with GPT2Pretokenizer (default) |
| `_train_impl(texts, vocab_size, pt)` | Train with specific pre-tokenizer |

---
## Roadmap

- Resume BPE training from existing vocabularies
- Unicode-native tokenization through custom `PreTokenizer`s
- Tokenizers specialized for new languages and domains

---

## Related projects

- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's BPE tokenizer (Rust + Python); the format and API mbpe is compatible with
- [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) — Rust port of tiktoken
- [minbpe](https://github.com/karpathy/minbpe) — Karpathy's minimal, from-scratch BPE reference implementation in pure Python




