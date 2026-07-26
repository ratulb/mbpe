# GAPS — What BPETokenizer[PT] Still Lacks vs. bpe.mojo

## 1. Context: Two BPE Projects

Two sibling Mojo BPE-tokenizer projects live side by side:

| | `simple_bpe/` (us) | `bpe.mojo/` (sibling) |
|---|---|---|
| Focus | Correctness-verified, clean architecture, fast encode/decode | Production features: pre-trained weight loading, standard formats |
| Pre-tokenization | 3 verified implementations (GPre, GPT-2 r50k_base, GPT-4 cl100k_base) | Hardcoded GPT-4 pattern only |
| Pre-tokenizer semantics | **First-match-wins** (correct regex alternation) | **Longest-match-wins** (incorrect for regex) |
| Architecture | Single parameterized `BPETokenizer[PT: PreTokenizer]` | Three-tier inheritance: `BasicTokenizer` → `RegexTokenizer` → `GPT4Tokenizer` |
| Decode speed | **109 M tok/s** (1.9× Rust tiktoken-rs) | Comparable (same memcpy-based flat storage) |
| Training speed | O(V×W) per merge — rescans every word | O(N) per merge — incremental pair stats |

The two projects are complementary. Our pre-tokenizers are **more correct** and our architecture is **cleaner**. But bpe.mojo is **more production-ready** — it loads real OpenAI weights, saves/loads in standard formats, handles special tokens, and trains faster.

---

## 2. bpe.mojo's Three-Tier Architecture

```
GPT4Tokenizer           ← loads pre-trained .tiktoken, adds byte_shuffle
    └── RegexTokenizer  ← adds Pretokenizer (GPT-4 regex) + special tokens
            └── BasicTokenizer  ← pure byte-level BPE (no pretokenization)
```

### BasicTokenizer
Pure byte-level BPE. No pretokenization, no special tokens. Train directly on raw bytes. This is the core merge engine.

### RegexTokenizer
Wraps `BasicTokenizer` + `Pretokenizer` (GPT-4 cl100k regex). Adds `register_special_tokens()` — splits on special-token boundaries before encoding. Training uses frequency-based merge selection (most-frequent-pair, not lowest-rank).

### GPT4Tokenizer
Wraps `RegexTokenizer` + `load_tiktoken()` (recover merges from base64 `.tiktoken` files) + `byte_shuffle` (permute byte→token ID to match GPT-4's rank ordering). Pre-trained only — `train()` raises an error.

---

## 3. Gap Analysis

### Gap 1: Incremental training stats (HIGH)

**Problem:** Our training loop recomputes all pair frequencies from scratch every merge:

```mojo
while len(self.vocab) < vocab_size:
    var pair_freqs = _compute_pair_freqs(splits, word_freqs)  # O(V×W)
    # pick best pair, merge...
```

This scans **every word × every token** in the corpus for each merge. With 500 merges and 475K tokens, that's **475K × 500 = ~238M token scans** during training.

**bpe.mojo's solution:** Incremental pair-stats update. When merging `(a,b) → new_id` at position `i`:

```
Old sequence:  [..., prev, a, b, next, ...]
New sequence:  [..., prev, new_id, next, ...]

Pairs destroyed:  (prev,a), (a,b), (b,next)  →  decrement from stats
Pairs created:    (prev,new_id), (new_id,next)  →  increment in stats
```

Only 5 pair-count operations per merge occurrence. **O(N) per merge** (N = occurrences of the merged pair) instead of **O(V×W)** (all tokens in all words).

**Implementation details from bpe.mojo (`basic_tokenizer.mojo:158–200`):**

```mojo
var new_ids = List[Int](capacity=len(ids))
var i = 0
while i < len(ids):
    if ids[i] == max_pair.first and ids[i + 1] == max_pair.second:
        # Decrement old pairs
        if len(new_ids) > 0:          # (prev, a)
            stats.decrement(new_ids[-1], ids[i])
        stats.decrement(ids[i], ids[i + 1])     # (a, b)
        if i + 2 < len(ids):          # (b, next)
            stats.decrement(ids[i + 1], ids[i + 2])
        # Increment new pairs
        if len(new_ids) > 0:          # (prev, new_id)
            stats.increment(new_ids[-1], new_id)
        if i + 2 < len(ids):          # (new_id, next)
            stats.increment(new_id, ids[i + 2])
        new_ids.append(new_id)
        i += 2
    else:
        new_ids.append(ids[i])
        i += 1
```

**Supporting structure:** `PairIntMap` (`token_map.mojo:62–158`) encodes pairs as `(first << 20) | second` for a single Int key, avoiding custom hash overhead. O(V) scan to find max pair.

**What to change:**
1. Replace our `Dict`-based pair counting with a `PairIntMap`-like structure
2. Store a flat `List[Int]` token sequence for each word (we already have `Dict[String, List[Int]]` splits)
3. Rewrite the merge loop to do incremental updates instead of recomputing pair_freqs from scratch
4. Keep the rest of the training pipeline (base vocab init, word splitting, etc.) unchanged

---

### Gap 2: Standard `.tiktoken` format (HIGH)

**Current state:** Our tokenizer saves/loads via JSON + Python interop:

```mojo
var json = Python.import_module("json")
json.dump(...)  # depends on Python being installed
```

**Problem:** This means:
- Python must be installed to save or load a tokenizer
- The JSON format is non-portable — no other BPE library understands it
- Base64 encoding (used by `.tiktoken`) is more compact and is the standard

**What `.tiktoken` looks like:**

```
0  base64_encode(bytes([0]))
1  base64_encode(bytes([1]))
...
100257 base64_encode(bytes([...some merge bytes...]))
```

Each line is `rank base64_bytes`. The base token entries define the byte shuffle, and the merge entries encode the full merge history.

**What we need:**
1. A pure-Mojo base64 encoder/decoder (no `base64` Python module)
2. A `.tiktoken` serializer that writes our merges in rank order
3. A `.tiktoken` parser that rebuilds merge rules from rank-ordered entries
4. Optionally: minbpe `.model` format support for compatibility with Andrej Karpathy's minbpe

---

### Gap 3: Special tokens (HIGH)

**What special tokens are:**

Special tokens are reserved token IDs that bypass BPE entirely. They're used as control signals in LLM chat templates:

```
<|endoftext|>   — marks end of a text segment
<|im_start|>    — start of an interleaved-message
<|im_end|>      — end of an interleaved-message
<|fim_prefix|>  — fill-in-the-middle prefix
<|fim_middle|>  — fill-in-the-middle middle
<|fim_suffix|>  — fill-in-the-middle suffix
```

**How bpe.mojo handles them (`regex_tokenizer.mojo:61–73`):**

Registration stores two maps:
```mojo
var special_tokens: Dict[String, Int]     # text → ID
var inverse_special: Dict[Int, String]     # ID → text
```

During `encode()` (lines 221–294):
1. Walk the input byte-by-byte
2. At each position, check if any special token starts here
3. If a match is found, emit the special token ID and skip past it
4. If no match, scan ahead to the next special-token boundary, BPE-encode the segment in between, and emit its IDs
5. Continue

**Implication for us:** Without this, `<|endoftext|>` in input text will be decomposed into `<`, `|`, `e`, `n`, `d`, ... and merged with neighbors. The round-trip `encode → decode` would lose the special boundary. Chat-style inputs would be corrupted.

**What to change:**
1. Add `special_tokens` and `inverse_special` fields to `BPETokenizer`
2. Add `register_special_tokens()` method
3. Modify `encode()` to split on special-token boundaries before BPE
4. Modify `decode()` to emit special token text for reserved IDs
5. Guard: normal BPE merges must never produce a special token ID

---

### Gap 4: Load pre-trained weights (HIGH)

This is the superset of Gaps 2 + 3 + byte shuffling. Loading a real GPT-4 tokenizer requires:

1. **Parse `.tiktoken`** — read base64-encoded entries (covers Gap 2)
2. **Recover merges** — entries in `.tiktoken` are rank-ordered; each entry is the result of a merge. Recover the two child IDs from the bytes of each entry.
3. **Build byte shuffle** — the base byte entries (ranks 0–255 in the tiktoken ordering) define the `byte_shuffle` table (see Gap 4a below)
4. **Register special tokens** — e.g., `o200k_base` reserves IDs like 100257+ for specials (covers Gap 3)
5. **Initialize vocab** — build decode strings for all tokens (with shuffled byte → original byte mapping)

**Why byte shuffle exists:**

OpenAI's `.tiktoken` files don't assign byte values to token IDs sequentially. In GPT-4's `o200k_base`:

- Byte `0x00` might be at rank 188
- Byte `'A'` (0x41) might be at rank 13
- etc.

This "shuffle" (`byte_shuffle: List[Int]` mapping original byte → token ID) exists because:
- The `.tiktoken` file specifies byte-token entries at whatever ranks the model was trained with
- These ranks are the "name" of each byte in the merge ranking system
- Without the shuffle, `mergeable_ranks[key]` would produce token IDs that don't match the training data

**How bpe.mojo builds the shuffle (`gpt4_tokenizer.mojo:287–295`):**

```mojo
for i in range(256):
    var single = List[UInt8](capacity=1)
    single.append(UInt8(i))
    var key = self._bytes_key(single)  # base64-encode 1 byte
    if key in mergeable_ranks:
        var shuffled_id = mergeable_ranks[key]
        self.byte_shuffle[i] = shuffled_id
        self.inverse_byte_shuffle[shuffled_id] = i
```

**Encoding with shuffle (`gpt4_tokenizer.mojo:319–356`):**

Each raw input byte must be looked up through `byte_shuffle` before entering BPE:

```mojo
var shuffled_byte = self.byte_shuffle[Int(raw_byte)]
```

**Decoding with shuffle (`gpt4_tokenizer.mojo:369–417`):**

Each vocab-entry byte must be run through `inverse_byte_shuffle` to restore the original value:

```mojo
var orig_byte = self.inverse_byte_shuffle[Int(shuffled_byte)]
```

---

### Gap 5: Python-free save/load (MEDIUM)

Currently our `save()` and `load()` depend on `from std.python import Python` for JSON serialization. Fixing this requires either:

- A pure-Mojo JSON encoder/decoder (std library or hand-written)
- Or switching to `.tiktoken` format (which already has a simple line-based format)

This gap is naturally closed by implementing Gap 2 (`.tiktoken` format), since `.tiktoken` is plain text with base64 entries and needs no JSON.

---

### Gap 6: Verbose training mode (LOW)

A `verbose=True` flag during training that prints merge progress. Straightforward — just add print statements gated by a flag.

---

## 4. What We Already Do Better

| Aspect | Us (simple_bpe) | bpe.mojo |
|---|---|---|
| **Pre-tokenizer correctness** | First-match-wins (correct regex alternation) | Longest-match-wins (incorrect — see PRETOKENIZER.md) |
| **Pre-tokenizer variants** | 3 (Ġ, r50k_base, cl100k_base) | 1 (cl100k_base, hardcoded) |
| **Pre-tokenizer verification** | Level A (split alignment) + Level B (end-to-end token IDs) both verified against Python regex | No published verification |
| **Architecture** | Single parameterized struct `BPETokenizer[PT]` — compile-time dispatch, no inheritance | Three-level inheritance — more complex, runtime dispatch |
| **Decode performance** | **109 M tok/s** (1.9× Rust tiktoken-rs) | Comparable but no published numbers vs Rust |
| **Encoding performance** | **10.9 M tok/s** (3.5× Rust tiktoken-rs) | Lower (byte-shuffle adds per-byte indirection) |

## 5. Priority Roadmap

```
Priority  Gap                          Effort    Impact
────────────────────────────────────────────────────────
 1        Incremental training stats   2–3 days  Training: O(V×W) → O(N)
 2        .tiktoken format             2–3 days  Interoperability, no-Python save/load
 3        Special tokens               1 day     Chat-format tokenization
 4        Load pre-trained weights     2–3 days  Use real GPT-4/4o/o200k_base
 5        Python-free JSON             0.5 day   (blocked on std library or .tiktoken)
 6        Verbose training             <0.5 day  Debug convenience
```

**Recommended order:** 1 → 2 → 3 → 4 (2 enables 4, 3 is prerequisite for 4).

Step 1 (incremental stats) is the biggest performance win and is purely internal — no formats, no external deps, no new files. It makes everything downstream faster and sets the foundation for the rest.
