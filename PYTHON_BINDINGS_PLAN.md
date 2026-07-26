# Python Bindings Plan

## Goals

1. Expose `BPETokenizer` as a Python extension module (`.so`) callable from `import simple_bpe`
2. Mirror the tiktoken API surface so `simple_bpe` feels familiar to users switching from OpenAI
3. Support all 3 pre-tokenizer variants (GPre, GPT2, GPT4) from Python
4. Allow both **loading pre-trained encodings** (via `.tiktoken` files) and **training from scratch**

---

## Target API (Python)

```python
import simple_bpe

# ── Load pre-trained (mirrors tiktoken.get_encoding) ──
enc = simple_bpe.get_encoding("gpt2")       # GPT-2 / r50k_base
enc = simple_bpe.get_encoding("cl100k")      # GPT-4 / cl100k_base
enc = simple_bpe.get_encoding("o200k")       # GPT-4o / o200k_base

# ── Training from scratch ──
enc = simple_bpe.train(
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
enc2 = simple_bpe.BPETokenizer.load("my_tokenizer.json")

enc.save_tiktoken("my_tokenizer.tiktoken")
enc2 = simple_bpe.BPETokenizer()
enc2.load_tiktoken("my_tokenizer.tiktoken")

# ── Special tokens ──
enc.register_special_tokens({"<|pad|>": 50000})
```

---

## Mojo Architecture

### Challenge

`BPETokenizer` is parameterized by a compile-time `PreTokenizer` type:

```mojo
struct BPETokenizer[PT: PreTokenizer = GPreTokenizer](Sized & Movable): ...
```

Python cannot supply Mojo type parameters at runtime. We need one Python-visible
type per PT variant, plus a factory.

### Solution: 3 Python-visible types + factory functions

| Python class | Mojo specialization |
|---|---|
| `GPreTokenizer`  | `BPETokenizer[GPreTokenizer]` |
| `GPT2Tokenizer`  | `BPETokenizer[GPT2Pretokenizer]` |
| `GPT4Tokenizer`  | `BPETokenizer[GPT4Pretokenizer]` |

All 3 share identical method bindings. A factory function `get_encoding()`
returns the right instance.

The user **can** construct any variant directly:

```python
from simple_bpe import GPreTokenizer, GPT2Tokenizer, GPT4Tokenizer

tok = GPT2Tokenizer()
tok.train(["hello world"], vocab_size=500)
```

But the recommended entry point is `get_encoding()` / `train()`.

### File layout

```
bpe_python/
├── __init__.py         # Python-side: imports from ._core
├── _core.mojo          # PyInit entry point, module builder, helpers
├── bindings.mojo       # Method binding helpers shared by all 3 types
├── gpre_bpe.mojo       # alias: type GPreBPETokenizer = BPETokenizer[GPreTokenizer]
├── gpt2_bpe.mojo       # alias: type GPT2BPETokenizer = BPETokenizer[GPT2Pretokenizer]
├── gpt4_bpe.mojo       # alias: type GPT4BPETokenizer = BPETokenizer[GPT4Pretokenizer]
└── py_init.mojo        # Actual Python-visible __init__ wrappers
```

### Python-visible method signatures

Each method is a `@staticmethod` that takes `py_self: PythonObject`
(or `self_ptr: UnsafePointer[Self]`) plus Python args, and returns `PythonObject`.

#### Constructor

```mojo
struct GPreBPETokenizer(BPETokenizer[GPreTokenizer]):

    @staticmethod
    def py_init(
        out self: Self,
        args: PythonObject,
        kwargs: PythonObject,
    ) raises:
        # Pretokenizer is fixed at compile time.
        # Optional kwargs: name (str)
        var name = "gpre"
        if kwargs.contains("name"):
            name = String(kwargs["name"])
        self = Self()
```

#### Core methods

Each of 3 types gets identical bindings:

```mojo
@staticmethod
def py_train(
    self_ptr: UnsafePointer[mut=True, Self],
    corpus_obj: PythonObject,       # list[str]
    vocab_size_obj: PythonObject,   # int
) raises:
    var corpus_list = Python.list(corpus_obj)
    var corpus = List[String]()
    for i in range(len(corpus_list)):
        corpus.append(String(corpus_list[i]))
    var vocab_size = Int(vocab_size_obj)
    self_ptr[].train(corpus, vocab_size)

@staticmethod
def py_encode(
    py_self: PythonObject,
    text_obj: PythonObject,
    kwargs: OwnedKwargsDict[PythonObject],
) raises -> PythonObject:
    var text = String(text_obj)
    # handle allowed_special / disallowed_special kwargs
    var ids = py_self.downcast_value_ptr[Self][].encode(text)
    return PythonObject(ids)  # Mojo List[Int] → Python list[int]

@staticmethod
def py_encode_ordinary(
    py_self: PythonObject,
    text_obj: PythonObject,
) raises -> PythonObject:
    var text = String(text_obj)
    var ids = py_self.downcast_value_ptr[Self][].encode_ordinary(text)
    return PythonObject(ids)

@staticmethod
def py_decode(
    py_self: PythonObject,
    tokens_obj: PythonObject,
) raises -> PythonObject:
    var py_list = Python.list(tokens_obj)
    var ids = List[Int]()
    for i in range(len(py_list)):
        ids.append(Int(py_list[i]))
    var result = py_self.downcast_value_ptr[Self][].decode(ids)
    return PythonObject(result)

@staticmethod
def py_len(py_self: PythonObject) raises -> PythonObject:
    var n = len(py_self.downcast_value_ptr[Self][])
    return PythonObject(n)

@staticmethod
def py_n_vocab(py_self: PythonObject) raises -> PythonObject:
    var n = len(py_self.downcast_value_ptr[Self][])
    return PythonObject(n)
```

#### Factory functions (module-level)

```mojo
def py_get_encoding(
    name_obj: PythonObject,
) raises -> PythonObject:
    var name = String(name_obj)
    if name == "gpt2":
        var tok = BPETokenizer[GPT2Pretokenizer]()
        tok.load_tiktoken("path/to/gpt2.tiktoken")  # bundled
        return PythonObject(alloc=tok^)
    elif name == "cl100k":
        ...
    elif name == "o200k":
        ...
    else:
        raise Error("unknown encoding: " + name)

def py_train(
    texts_obj: PythonObject,
    vocab_size_obj: PythonObject,
    kwargs: OwnedKwargsDict[PythonObject],
) raises -> PythonObject:
    var pretokenizer = "gpre"
    if kwargs.contains("pretokenizer"):
        pretokenizer = String(kwargs["pretokenizer"])
    # ... dispatch to right PT variant ...
```

### Module builder

```mojo
@export
def PyInit_simple_bpe() abi("C") -> PythonObject:
    try:
        var mb = PythonModuleBuilder("simple_bpe")

        # Bind all 3 tokenizer types
        _ = mb.add_type[GPreBPETokenizer]("GPreTokenizer") \
            .def_py_init[GPreBPETokenizer.py_init]() \
            .def_method[GPreBPETokenizer.py_train]("train") \
            .def_method[GPreBPETokenizer.py_encode]("encode") \
            .def_method[GPreBPETokenizer.py_encode_ordinary]("encode_ordinary") \
            .def_method[GPreBPETokenizer.py_decode]("decode") \
            .def_method[GPreBPETokenizer.py_len]("__len__") \
            .def_method[GPreBPETokenizer.py_n_vocab]("n_vocab") \
            .def_staticmethod[GPreBPETokenizer.py_load]("load")
        # Same for GPT2BPETokenizer, GPT4BPETokenizer...

        # Bind factory functions
        mb.def_function[py_get_encoding]("get_encoding")
        mb.def_function[py_train]("train")
        mb.def_function[py_list_encodings]("list_encodings")

        return mb.finalize()
    except e:
        abort("error creating simple_bpe module:", e)
```

---

## Comparison: tiktoken ↔ simple_bpe Python API

| tiktoken | simple_bpe | Notes |
|---|---|---|
| `tiktoken.get_encoding("gpt2")` | `simple_bpe.get_encoding("gpt2")` | Same pattern |
| `tiktoken.encoding_for_model("gpt-4")` | `simple_bpe.encoding_for_model(...)` | Future |
| `enc.encode(text)` | `enc.encode(text)` | Same |
| `enc.encode_ordinary(text)` | `enc.encode_ordinary(text)` | Same |
| `enc.decode(tokens)` | `enc.decode(tokens)` | Same |
| `enc.decode_bytes(tokens)` | `enc.decode_bytes(tokens)` | Future |
| `enc.n_vocab` | `enc.n_vocab` | Same |
| `enc.name` | `enc.name` | Same |
| `enc.special_tokens_set` | `enc.special_tokens_set` | Future |
| `enc.register_special_tokens(...)` | `enc.register_special_tokens(...)` | New |
| — | `enc.train(texts, vocab_size)` **New** | Not in tiktoken |
| — | `simple_bpe.train(texts, vsize, pt)` | **New** |
| — | `enc.merges` | **New** |

simple_bpe adds training APIs that tiktoken does not expose.

---

## Implementation Phases

### Phase 1 — Minimal v0 (this PR)

- Single file: `simple_bpe.mojo`
- Bind `BPETokenizer[GPreTokenizer]` only (default PT)
- Expose: `__init__`, `train`, `encode`, `encode_ordinary`, `decode`, `__len__`, `n_vocab`, `save`, `load`
- Module-level: `get_encoding(name)` loads from bundled `.tiktoken`
- Build: `mojo build simple_bpe.mojo --emit shared-lib -o simple_bpe.so`

### Phase 2 — Full PT support

- Bind `GPT2Tokenizer` and `GPT4Tokenizer`
- Module-level `train(texts, vocab_size, pretokenizer=...)` factory
- `register_special_tokens`, `save_tiktoken`, `load_tiktoken`

### Phase 3 — tiktoken API parity

- `decode_bytes`, `decode_single_token_bytes`, `token_byte_values`
- `encode_single_token`
- `name` property
- `decode_with_offsets`
- Batch methods (`encode_batch`, `decode_batch`)

---

## Build & Distribution

```bash
# Build extension module
mojo build bpe_python/simple_bpe.mojo \
    --emit shared-lib \
    -o simple_bpe.so

# Use from Python
export PYTHONPATH=.
python -c "import simple_bpe; enc = simple_bpe.get_encoding('gpt2')"
```

Future: distribute via PyPI using `mojo build` in `setup.py`.

---

## Known Limitations (current Mojo Python interop)

1. No computed properties — must use `get_n_vocab()` or `__len__` pattern
2. No `**kwargs` syntax — must use `OwnedKwargsDict[PythonObject]`
3. No variadic `*args` — use `def_py_function` for variable-arity
4. Max 6 `PythonObject` arguments per bound function
5. Methods require `@staticmethod` with `py_self` / `self_ptr` boilerplate
6. Types need `Writable` trait for `add_type`; `Movable` for `def_py_init`
