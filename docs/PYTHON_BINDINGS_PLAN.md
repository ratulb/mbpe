# Python Bindings Plan

## Goals

1. Expose `BPETokenizer` as a Python extension module (`.so`) callable from `import mbpe`
2. Mirror the tiktoken API surface so `mbpe` feels familiar to users switching from OpenAI
3. Support all 3 pre-tokenizer variants (GPre, GPT2, GPT4) from Python
4. Allow both **loading pre-trained encodings** (via `.tiktoken` files) and **training from scratch**

---

## Target API (Python)

```python
import mbpe

# ── Load pre-trained (mirrors tiktoken.get_encoding) ──
enc = mbpe.get_encoding("gpt2")       # GPT-2 / r50k_base
enc = mbpe.get_encoding("cl100k")      # GPT-4 / cl100k_base
enc = mbpe.get_encoding("o200k")       # GPT-4o / o200k_base

# ── Training from scratch ──
enc = mbpe.train(
    texts=["hello world", "some other text"],
    vocab_size=1000,
    pretokenizer="gpt2",          # "gpre" | "gpt2" | "gpt4"
)

# ── Encode / Decode (tiktoken-compatible) ──
tokens = enc.encode("hello world")            # list[int]
tokens = enc.encode_ordinary("hello world")   # list[int], no special tokens
text   = enc.decode(tokens)                   # str
tokens = enc.encode("hello <|endoftext|>",
                    allowed_special={"<|endoftext|>"})

# ── Info ──
len(enc)                             # vocab size
enc.n_vocab                          # same
enc.name                             # "gpt2"
enc.merges                           # list of (a, b, merged_id) tuples

# ── Serialization ──
enc.save("my_tokenizer.json")
enc2 = mbpe.BPETokenizer.load("my_tokenizer.json")

enc.save_tiktoken("my_tokenizer.tiktoken")
enc2 = mbpe.BPETokenizer()
enc2.load_tiktoken("my_tokenizer.tiktoken")

# ── Special tokens ──
enc.register_special_tokens({"<|pad|>": 50000})
```

---

## Current API (Phase 1 — implemented)

```python
import mbpe

# GPreTokenizer only (other types registered for get_encoding)
tok = mbpe.GPreTokenizer()
tok.train(["hello world", "test"], vocab_size=300)
tok.encode("hello world")            # list[int]
tok.encode_ordinary("hello world")   # list[int]
tok.decode([264, 278])               # str
tok.n_vocab()                        # int
tok.save("tok.json")
tok2 = mbpe.GPreTokenizer.load("tok.json")

# Pre-trained encodings (no bundled .tiktoken files yet)
enc = mbpe.get_encoding("gpt2")
enc = mbpe.get_encoding("cl100k")
enc = mbpe.get_encoding("o200k")

# Also available: GPT2Tokenizer, GPT4Tokenizer, GPT4oTokenizer
```

---

## Implementation Architecture

### Challenge

`BPETokenizer` is parameterized by a compile-time `PreTokenizer` type:
```mojo
struct BPETokenizer[PT: PreTokenizer = GPreTokenizer](Sized & Movable): ...
```
Python cannot supply Mojo type parameters at runtime. We need one Python-visible
type per PT variant, plus a factory.

### Solution: 4 Python-visible types + factory function

| Python class | Mojo specialization |
|---|---|
| `GPreTokenizer`  | `BPETokenizer[GPreTokenizer]` |
| `GPT2Tokenizer`  | `BPETokenizer[GPT2Pretokenizer]` |
| `GPT4Tokenizer`  | `BPETokenizer[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]` |
| `GPT4oTokenizer` | `BPETokenizer[GPT4Pretokenizer[ByteMapping.SHUFFLED]]` |

All 4 types share identical method bindings. A wrapper-per-comptime-parameter
pattern avoids code duplication:

```mojo
# Parameterized template instantiated for each type via `alias`
def _train_impl[T: ImplicitlyDeletable & Writable & Movable](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[T]()
    ...

# Concrete aliases
alias _train_gpre = _train_impl[BPETokenizer[GPreTokenizer]]
alias _train_gpt2 = _train_impl[BPETokenizer[GPT2Pretokenizer]]
```

### File layout

Single file: `python-binding/mbpe.mojo` contains everything — type aliases,
parameterized wrappers, concrete `alias` instantiations, init/load functions,
and the `PyInit_mbpe` entry point.

### Module entry point

```mojo
@export
def PyInit_mbpe() abi("C") -> PythonObject:
    var mb = PythonModuleBuilder("mbpe")
    _ = mb.add_type[GPreTK]("GPreTokenizer").def_py_init[...]()...
    _ = mb.add_type[GPT2TK]("GPT2Tokenizer").def_py_init[...]()...
    _ = mb.add_type[GPT4TK]("GPT4Tokenizer").def_py_init[...]()...
    _ = mb.add_type[GPT4oTK]("GPT4oTokenizer").def_py_init[...]()...
    mb.def_function[py_get_encoding]("get_encoding")
    return mb.finalize()
```

---

## Comparison: tiktoken ↔ mbpe Python API

| tiktoken | mbpe | Notes |
|---|---|---|
| `tiktoken.get_encoding("gpt2")` | `mbpe.get_encoding("gpt2")` | Same pattern |
| `tiktoken.encoding_for_model("gpt-4")` | `mbpe.encoding_for_model(...)` | Future |
| `enc.encode(text)` | `enc.encode(text)` | Same |
| `enc.encode_ordinary(text)` | `enc.encode_ordinary(text)` | Same |
| `enc.decode(tokens)` | `enc.decode(tokens)` | Same |
| `enc.decode_bytes(tokens)` | `enc.decode_bytes(tokens)` | ✅ Done |
| `enc.n_vocab` | `enc.n_vocab()` | Method (not property, Mojo limitation) |
| `enc.encode_single_token(text)` | `enc.encode_single_token(text)` | ✅ Done |
| `enc.name` | `enc.name` | Same |
| `enc.special_tokens_set` | `enc.special_tokens_set` | Future |
| `enc.register_special_tokens(...)` | `enc.register_special_tokens(...)` | New |
| — | `enc.train(texts, vocab_size)` **New** | Not in tiktoken |
| — | `mbpe.train(texts, vsize, pt)` | **New** |
| — | `enc.merges` | **New** |

mbpe adds training APIs that tiktoken does not expose.

---

## Implementation Phases

### Phase 1 — Minimal v0 ✅ (done)

- Single file: `python-binding/mbpe.mojo`
- Parameterized wrapper-functions with `alias` instantiations
- Bind all 4 tokenizer types: GPreTokenizer, GPT2Tokenizer, GPT4Tokenizer, GPT4oTokenizer
- Expose: `__init__`, `train`, `encode`, `encode_ordinary`, `decode`, `n_vocab`, `name`, `save`, `load`
- Module-level: `get_encoding(name)` factory (loads from bundled `.tiktoken` files in `data/`)
- Build: `mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/mbpe.so`

### Phase 2 — Bundled encodings + training factory ✅ (done)

- `.tiktoken` files for gpt2 / cl100k / o200k in `data/`
- `get_encoding()` loads from bundled files
- Module-level `train(texts, vocab_size, pretokenizer=...)` factory
- `register_special_tokens`, `save_tiktoken`, `load_tiktoken`

### Phase 3 — tiktoken API parity ✅ (done)

- `decode_bytes`, `decode_single_token_bytes`, `token_byte_values`
- `encode_single_token`
- `name` property
- `decode_with_offsets`
- ❌ Batch methods (`encode_batch`, `decode_batch`) — not implemented

---

## Build & Distribution

```bash
# Build extension module
mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/mbpe.so

# Use from Python
export PYTHONPATH=python-binding
python -c "import mbpe; enc = mbpe.get_encoding('gpt2')"
```

Future: distribute via PyPI using `mojo build` in `setup.py`.

---

## Known Limitations (current Mojo Python interop)

1. No computed properties — must use `n_vocab()` method, not `n_vocab` attribute
2. `len(enc)` not supported — `__len__` magic method isn't wired to `sq_length` slot
3. No `**kwargs` syntax — `def_py_method` uses `METH_VARARGS` signature
4. Methods require `PythonObject`-based wrappers with `downcast_value_ptr` boilerplate
5. Types need `Writable` trait for `add_type`; `ImplicitlyDeletable` & `Movable` for `def_py_init`
6. `alias` of parameterized functions required because `def_py_method` needs concrete function references
