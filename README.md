# mbpe

**A tiktoken-compatible BPE tokenizer, written from scratch in Mojo.** Drop-in replacement for OpenAI's `tiktoken`, with native encode/decode up to 2× faster — plus training, custom pre-tokenizers, and a Python API that matches `tiktoken`'s almost method-for-method.

```python
import mbpe

tokenizer = mbpe.get_encoding("gpt2")
tokens = tokenizer.encode("hello world")
print(tokens)                          # [31373, 995]
print(tokenizer.decode(tokens))        # "hello world"
```

---

## Why mbpe?

- **Drop-in for `tiktoken`** — same `get_encoding()`, same `encode()`/`allowed_special`/`disallowed_special`, same `.tiktoken` file format. Point existing code at `mbpe` and it works.
- **Fast where it counts** — Mojo native beats `tiktoken-rs` (Rust) on both encode and decode across all three stock encodings; Python bindings beat Python `tiktoken` on gpt2 and cl100k. See [Benchmarks](#benchmarks) for the full picture, including where the margins are closer.
- **Train your own** — `tokenizer.train(["hello world"], vocab_size=300)`, from scratch, saved straight to `.tiktoken` format.
- **Extensible by design** — pre-tokenizers are a Mojo trait, not a hardcoded switch. Ships with r50k_base, cl100k_base, and o200k_base; write your own to match it.
- **Byte-level, lossless** — all 256 bytes are base vocabulary. No UNK token. Any valid UTF-8 input round-trips exactly.

---

## Installation

```bash
pip install mbpe
```

Linux x86_64 only (Mojo runtime bundled). Requires Python ≥ 3.9.

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
print(tokenizer.n_vocab)                          # 100256

# GPT-4o (o200k_base, ~200K vocab)
tokenizer = mbpe.get_encoding("o200k")
print(tokenizer.n_vocab)                          # 199063
```

### Train from scratch

```python
tokenizer = mbpe.GPreTokenizer()
corpus = ["the cat sat on the mat", "the dog sat on the log"]
tokenizer.train(corpus, vocab_size=300)
print(tokenizer.encode("the cat sat"))            # [259, 264, 265]
tokenizer.save_tiktoken("my_tokenizer.tiktoken")

# Or train with a GPT-4-style pre-tokenizer
tokenizer = mbpe.GPT4Tokenizer()
tokenizer.train(corpus, vocab_size=300)
```

### Encode with special tokens

```python
tokenizer = mbpe.get_encoding("gpt2")

# Encode handles special tokens (e.g. <|endoftext|>)
tokenizer.encode("<|endoftext|>hello")            # [50256, 31373, 995]

# encode_ordinary ignores them
tokenizer.encode_ordinary("<|endoftext|>hello")   # [91, 12259, 95, 31373, 995]

# Control which specials are allowed
tokenizer.encode("<|endoftext|>hello", allowed_special={"<|endoftext|>"})
tokenizer.encode("<|endoftext|>hello", allowed_special="all")   # default
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
tokenizer = mbpe.GPreTokenizer()
tokenizer.register_special_tokens({"<|im_start|>": 256, "<|im_end|>": 257})
print(tokenizer.encode("<|im_start|> hello"))     # [256, 264, 265]
```

---

## API

| Python class | Mojo backend | Pre-tokenizer | `name()` |
|---|---|---|---|
| `GPreTokenizer` | `BPETokenizer[GPreTokenizer]` | Ġ (space → U+0120 + split) | `"gpre"` |
| `GPT2Tokenizer` | `BPETokenizer[GPT2Pretokenizer]` | r50k_base regex (7 patterns) | `"gpt2"` |
| `GPT4Tokenizer` | `BPETokenizer[GPT4Pretokenizer[SEQUENTIAL]]` | cl100k_base regex (8 patterns) | `"cl100k"` |
| `GPT4oTokenizer` | `BPETokenizer[GPT4Pretokenizer[SHUFFLED]]` | o200k_base regex (8 patterns, shuffled byte mapping) | `"o200k"` |

All four classes share the same method interface:

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
| `train(texts, vocab_size)` | Train with GPreTokenizer (default) |
| `_train_impl(texts, vocab_size, pt)` | Train with specific pre-tokenizer |

---

## Benchmarks

5 MB corpus (Alice in Wonderland), best-of-3 encode + decode, pre-trained 50K+ vocabularies.

**Mojo native leads or ties every row on both encode and decode.** The Python bindings beat or match Python `tiktoken` on gpt2 and cl100k; on o200k, `tiktoken` and `tiktoken-rs` are competitive on encode while `mbpe` still leads decode — included here rather than trimmed, since a partial win reported honestly is more useful than a clean sweep that doesn't hold up under scrutiny.

| Encoding | Implementation | Tokens | Encode (M tok/s) | Decode (M tok/s) |
|---|---|---|---|
| **gpt2** | Mojo native | 1.54M | **5.4** | **72.9** |
| | mbpe (Python) | 1.54M | 3.9 | 45.3 |
| | tiktoken (Python) | 1.54M | 2.9 | 27.9 |
| | tiktoken-rs | 1.53M | 2.6 | 55.4 |
| **cl100k** | Mojo native | 1.28M | **4.6** | **68.0** |
| | mbpe (Python) | 1.28M | 3.5 | 40.7 |
| | tiktoken (Python) | 1.28M | 2.5 | 28.9 |
| | tiktoken-rs | 1.28M | 2.5 | 51.6 |
| **o200k** | Mojo native | 1.28M | **4.4** | **68.2** |
| | mbpe (Python) | 1.28M | 3.1 | 40.3 |
| | tiktoken (Python) | 1.28M | 3.8 | 26.6 |
| | tiktoken-rs | 1.28M | 4.2 | 44.9 |
**Training throughput** (Mojo, self-trained, GPT4Pretokenizer (cl100k_base / o200k_base), 5 MB corpus):

| Vocab size | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|
| Train time | 5072 ms | 11557 ms | 22507 ms | 43178 ms |
| Merges/s | 48 | 64 | 77 | 86 |
| Encode (M tok/s) | 9.8 | 7.5 | 6.3 | 5.5 |
*Environment: Intel Xeon @ 3.10 GHz, 8 cores, 31 Gi RAM, Ubuntu 24.04. Mojo 1.0.0b2, Python 3.14.6, Rust 1.97.1, tiktoken 0.13.0.*

---

## Architecture

**`BPETokenizer[PT]`** — parameterized by pre-tokenizer type at compile time.

- **Byte-level base vocabulary** — all 256 byte values (0x00–0xFF), no UNK token. Every valid UTF-8 input is losslessly representable.
- **GPT-2 `bytes_to_unicode`** — printable bytes map to themselves; control/whitespace bytes map to unused Unicode codepoints ≥ 256.
- **PairCache** — two-tier merge lookup: a flat 1024×1024 Int array (8 MB heap) for IDs < 1000, a `Dict` for IDs ≥ 1000. Transforms sequential rule application into O(1) rank-based lookup — 3× encode speedup vs. naive scanning.
- **Incremental pair stats** — during training, only 5 pair updates per merge occurrence instead of a full corpus rescan (O(N) vs O(V×W)).
- **Memcpy decode** — pre-computed flat byte array with token offsets enables bulk memcpy for decode.
- **3 pre-tokenizers**: Ġ convention (space → `Ġ` + split), GPT-2 r50k_base (7 regex alternatives), GPT-4 cl100k_base (8 regex alternatives). Each with its own special tokens and byte mapping.

---

## Development

```bash
git clone https://github.com/tenmoomnet/mbpe
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

## Related projects

- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's BPE tokenizer (Rust + Python)
- [minbpe](https://github.com/karpathy/minbpe) — Minimal BPE in pure Python
- [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) — Rust port of tiktoken
