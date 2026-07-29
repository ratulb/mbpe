# mbpe

[![Tests](https://github.com/ratulb/mbpe/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ratulb/mbpe/actions/workflows/python-tests.yml)

A **high-performance**, **trainable**, **tiktoken-compatible BPE tokenizer** written in **Mojo**.
The compile-time `PreTokenizer` trait enables **GPT-2, GPT-4, GPT-4o** and custom tokenization pipelines without changing the core tokenizer.

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
- **Fast Python bindings** - Substantially outperform Python `tiktoken` and match or exceed `tiktoken-rs` across the supported OpenAI encodings.
- **Train your own** — `tokenizer.train(["hello world"], vocab_size=300)`, then save directly to `.tiktoken` format.
- **Extensible by design** — `PreTokenizer` is a Mojo trait, not a hardcoded implementation. Ships with r50k_base, cl100k_base, and o200k_base; write your own to match it.
- **Byte-level, lossless** — all 256 bytes are base vocabulary. No UNK token. Any valid UTF-8 input round-trips exactly.

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
tokenizer = mbpe.GPreTokenizer()
corpus = ["the cat sat on the mat", "the dog sat on the log"]
tokenizer.train(corpus, vocab_size=300)
print(tokenizer.encode("the cat sat"))            # [259, 270, 265]
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

### Load an encoding

BPETokenizer[PT] / Tokenizers.get[PT] → returns comptime parameterized tokenizer. Loads from `.tiktoken` files located via `MBPE_DATA_DIR` env var, falling back to `./data/`

```mojo

from bpe.tokenizer import Tokenizers

var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
var ids = gpt2.encode("hello world")
print(gpt2.decode(ids))

```
### Train:

```mojo
from bpe.tokenizer import BPETokenizer
from bpe.pretokenizer import GPT2Pretokenizer

var tok = BPETokenizer[GPT2Pretokenizer]()
tok.train((["hello world"]), 300)
```


---

## Benchmarks

5 MB corpus (Alice in Wonderland), best-of-3 encode + decode, pre-trained 50K+ vocabularies.

> **Across all three OpenAI encodings, native Mojo is consistently the fastest implementation for both encoding and decoding, while the Python bindings substantially outperform Python tiktoken**.

#### gpt2 (r50k_base)

| Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|---|
| **mbpe — Mojo native** | 1.54M | **6.9** | **110.1** |
| mbpe — Python bindings | 1.54M | 6.2 | 61.6 |
| tiktoken (Python) | 1.54M | 2.8 | 27.2 |
| tiktoken-rs | 1.53M | 2.5 | 54.1 |

#### cl100k (cl100k_base)

| Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|---|
| **mbpe — Mojo native** | 1.28M | **5.6** | **102.5** |
| mbpe — Python bindings | 1.28M | 4.7 | 51.6 |
| tiktoken (Python) | 1.28M | 2.3 | 28.7 |
| tiktoken-rs | 1.28M | 2.5 | 51.6 |

#### o200k (o200k_base)

| Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|---|
| **mbpe — Mojo native** | 1.28M | **4.6** | **97.4** |
| mbpe — Python bindings | 1.28M | 4.3 | 54.7 |
| tiktoken (Python) | 1.28M | 3.6 | 26.3 |
| tiktoken-rs | 1.28M | 4.0 | 45.1 |

**Training throughput** (Mojo, self-trained, GPT4Pretokenizer (cl100k_base / o200k_base), 5 MB corpus):

| Vocab size | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|
| Train time | 5271 ms | 12204 ms | 24230 ms | 45200 ms |
| Merges/s | 46 | 60 | 71 | 82 |
| Encode (M tok/s) | 10.8 | 8.4 | 6.8 | 5.9 |

*Environment: Intel Xeon @ 3.10 GHz, 8 cores, 31 Gi RAM, Ubuntu 24.04. Mojo 1.0.0b2, Python 3.14.6, Rust 1.97.1, tiktoken 0.13.0.*

---

## Architecture

At the core of mbpe is `BPETokenizer[PT]`, where `PT` is any implementation of the compile-time `PreTokenizer` trait.

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


> A `PreTokenizer` converts input text into a sequence of symbols (e.g. regex chunks, Ġ-prefixed words, or future custom segmentations) before BPE merging begins.                                          


### comptime aliases:

- `Tokenizers.get[Tokenizers.gpt2]()`  → `BPETokenizer[GPT2Pretokenizer]` → r50k_base
- `Tokenizers.get[Tokenizers.cl100k]()` → `BPETokenizer[GPT4Pretokenizer[SEQUENTIAL]]` → cl100k_base
- `Tokenizers.get[Tokenizers.o200k]()`  → `BPETokenizer[GPT4Pretokenizer[SHUFFLED]]` → o200k_base


---

## Design highlights

- Compile-time `PreTokenizer` trait
- O(1) merge lookup (`MergeLookup`)
- Incremental pair statistics for training
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

# Run Mojo tests (78 total)
# pixi run mojo main.mojo            # 36 tests
# pixi run mojo -I . tests/test_tokenizer.mojo           # 9 tests
# pixi run mojo -I . tests/exhaustive_tokenizer.mojo     # 33 tests

# Run Python tests
# pixi run --environment dev python -m pytest tests/python/ -v
```

---

## Roadmap

- Resume BPE training from existing vocabularies
- Unicode-native tokenization through pluggable pre-tokenizers
- Tokenizers specialized for new languages and domains

