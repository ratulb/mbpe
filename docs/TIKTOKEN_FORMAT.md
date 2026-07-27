# `.tiktoken` Format — Design & Implementation Plan

## 1. What `.tiktoken` Is

OpenAI's standard distribution format for pre-trained BPE tokenizer vocabularies
(GPT-2, GPT-4, GPT-4o, etc.).  A plain-text file, one token per line:

```
<base64(token_bytes)> <rank>\n
```

- `token_bytes` — raw UTF-8 bytes of the token as seen by BPE (not display strings)
- `rank` — token ID (0–255 for single bytes, 256+ for merged tokens)
- No headers, no footers.  Lines separated by `\n`.  Empty lines skipped.
- Compatible with Python `tiktoken`, Rust `tiktoken-rs`, and bpe.mojo.

**File example (`o200k_base.tiktoken`, 199,998 lines):**
```
IQ== 0         # byte 0x00 -> base64([0x00])
Ig== 1         # byte 0x01
...
aWZmZW4= 199993
IHBhdHJvbmVz 199994
```

- Rank 0 is always byte `0x00`, rank 1 is byte `0x01`, ..., rank 255 is byte `0xFF`
- Ranks 256+ are merge tokens (byte sequences > 1 byte)

---

## 2. What We Need

### 2a. Save — write `.tiktoken` from trained state

Current `save()` uses Python `json` → non-portable.  We need:

```
for rank in 0..vocab_size-1:
    raw_bytes = cp_to_byte each codepoint in vocab[rank]
    encoded = b64encode(raw_bytes)
    file.write(encoded + " " + rank + "\n")
```

**Key: reconstruct raw bytes from display strings.**  Our `vocab[rank]` stores
display strings (safe Unicode codepoints via GPT-2's `bytes_to_unicode` table).
Each codepoint maps back to a raw byte (0–255) through `cp_to_byte`.

For single-byte tokens (ranks 0–255) this is always one codepoint → one byte.
For merge tokens (ranks 256+) it's the concatenation of the children's bytes.

### 2b. Load — rebuild tokenizer state from `.tiktoken`

```
for each line:
    raw_bytes = b64decode(parts[0])
    rank = Int(parts[1])
    mergeable_ranks[bytes_key(raw_bytes)] = rank
    all_tokens[rank] = raw_bytes

recover merges from all_tokens (see §3)
rebuild internal state (see §4)
```

---

## 3. Merge Recovery Algorithm

`.tiktoken` stores only `(bytes, rank)` pairs — no explicit merge rules.
Merge rules must be recovered by decomposing each multi-byte token byte-by-byte
and finding which adjacent pair was merged last.

### 3a. `_bpe(mergeable_ranks, token_bytes, max_rank) → List[List[UInt8]]`

Decompose `token_bytes` into its BPE merge path, stopping before the final merge
that would produce a token with rank ≥ `max_rank`.

```
parts = List[List[UInt8]](capacity=len(token_bytes))
for each byte b in token_bytes:
    parts.append([b])

loop:
    min_idx = -1
    min_rank = -1
    for i in 0..len(parts)-1:
        concat = List[UInt8](capacity=len(parts[i]) + len(parts[i+1]))
        concat.append(parts[i]) ... concat.append(parts[i+1])
        key = _bytes_key(Span[UInt8](concat))
        if key in mergeable_ranks:
            rank = mergeable_ranks[key]
            if min_idx < 0 or rank < min_rank:
                min_idx = i; min_rank = rank
    if min_idx < 0 or (max_rank >= 0 and min_rank >= max_rank):
        break

    merged = List[UInt8](capacity=len(parts[min_idx]) + len(parts[min_idx+1]))
    ...copy both parts into merged...
    new_parts = List[List[UInt8]](capacity=len(parts) - 1)
    ...copy all parts except the two merged ones, insert merged at min_idx...
    parts = new_parts^

return parts
```

**Allocation strategy:**
- `parts` pre-allocated to `len(token_bytes)` slots
- Each `concat` pre-allocated to exact combined capacity (1 alloc per pair check)
- `new_parts` pre-allocated to `len(parts) - 1` (1 alloc per merge iteration)
- The `List[UInt8]` for each byte in the initial split is unavoidable (256 vocab)

**Result when token was formed by a single merge `(A,B) → token_id`:**
`parts = [A, B]` — exactly two elements.  `A` and `B` may themselves be
multi-byte tokens with lower ranks.

**Result when nested merges create ambiguity:** 3+ elements.  Scan all
adjacent pairs for the one with combined rank = `token_id` (the last merge
applied).  If no exact match, pick the pair with highest combined rank
still < `token_id`.

### 3b. `_recover_merges(mergeable_ranks, all_tokens)`

```
merges = []
for token_id = 256 to vocab_size-1:
    bytes = all_tokens[token_id]
    if len(bytes) <= 1: continue

    parts = _bpe(mergeable_ranks, bytes, token_id)

    if len(parts) == 2:
        left_id = mergeable_ranks[bytes_key(parts[0])]
        right_id = mergeable_ranks[bytes_key(parts[1])]
        merges.append(MergeRule(left_id, right_id, token_id))

    elif len(parts) > 2:
        # Nested merges: find the pair whose combined rank = token_id
        best_cr = -1; best_left = -1; best_right = -1
        for each adjacent pair (i, i+1):
            concat = parts[i] ++ parts[i+1]
            if bytes_key(concat) in mergeable_ranks:
                cr = mergeable_ranks[bytes_key(concat)]
                if cr <= token_id and cr > best_cr:
                    left_key = bytes_key(parts[i])
                    right_key = bytes_key(parts[i+1])
                    if left_key in mergeable_ranks
                       and right_key in mergeable_ranks:
                        lr = mergeable_ranks[left_key]
                        rr = mergeable_ranks[right_key]
                        if lr < token_id and rr < token_id:
                            best_cr = cr
                            best_left = lr
                            best_right = rr
        if best_left >= 0:
            merges.append(MergeRule(best_left, best_right, token_id))

return merges
```

### 3c. `_bytes_key(bytes_span) → String`

Builds a dict key from raw bytes for `mergeable_ranks` lookups:

```
key = ""
for i in 0..len(bytes_span):
    if i > 0: key += ","
    key += String(Int(bytes_span[i]))
return key
```

Example: `[97, 98, 99]` → `"97,98,99"`

### 3d. Why bytes_key and not `String(from_utf8=...)`?

The `.tiktoken` format stores raw bytes that may not be valid UTF-8
(e.g., byte value `0x00`, partial multi-byte sequences).  Using `bytes_key`
avoids UTF-8 validity issues and produces unique, deterministic keys.

---

## 4. Rebuilding Internal State

After merge recovery produces `merges: List[MergeRule]`:

### 4a. Vocab display strings

Each raw byte in a token must be mapped through GPT-2's `bytes_to_unicode`
table to get the safe display codepoint:

```
for token_id in 0..vocab_size-1:
    raw_bytes = all_tokens[token_id]
    display = String(capacity=len(raw_bytes) * 3)
    for b in raw_bytes:
        display += chr(byte_to_cp[Int(b)])
    vocab.append(display)
```

Pre-allocating `display` with `capacity=len(raw_bytes) * 3` avoids reallocation
as we append codepoints (safe codepoints 256+ encode to 2–3 UTF-8 bytes).

For single-byte tokens (0–255): always one codepoint, display = `chr(byte_to_cp[b])`.
For merge tokens (256+): multi-codepoint string.

This is safe because `byte_to_cp` maps every byte 0–255 to a valid Unicode
codepoint (printable bytes → themselves, non-printable → ≥ 256).

### 4b. Decode storage (`token_bytes`, `token_offsets`, `token_lengths`)

Build the flat byte array for memcpy decode, same pattern as `train()`:

```
for token_id in 0..vocab_size-1:
    display = vocab[token_id]
    token_offsets.append(len(token_bytes))
    pending = -1
    for cp in display.codepoints():
        b = cp_to_byte[Int(cp)]
        if b == 0xA0 and pending == 0xC4:
            token_bytes.append(0x20)  # Ġ→space substitution
            pending = -1
        else:
            if pending >= 0:
                token_bytes.append(UInt8(pending))
            pending = b
    if pending >= 0:
        token_bytes.append(UInt8(pending))
    token_lengths.append(len(token_bytes) - token_offsets[-1])
token_offsets.append(len(token_bytes))
```

The Ġ→0x20 substitution is the same heuristic used in `train()` and `load()`:
if any two consecutive byte values (via `cp_to_byte`) happen to be
`[0xC4, 0xA0]`, replace with `[0x20]`.  This is a decode-speed optimization
internal to our representation and is transparent to .tiktoken consumers.

### 4c. `merge_cache`

After `merges` is populated, rebuild the two-tier cache:

```
for merge in merges:
    merge_cache.set(merge.first, merge.second, merge.merged)
```

### 4d. `byte_to_cp` / `cp_to_byte`

These are always the same — GPT-2 `bytes_to_unicode` mapping, independent of
the loaded vocabulary.  Already initialized in `__init__`.

---

## 5. Interface

### 5a. `save_tiktoken(path: String)`

```mojo
def save_tiktoken(mut self, path: String) raises:
    with open(path, "w") as f:
        for token_id in range(self.vocab_size):
            var raw = List[UInt8]()
            var display = self.vocab[token_id]
            for cp in display.codepoints():
                raw.append(UInt8(self.cp_to_byte[Int(cp)]))
            var encoded = b64encode(Span[UInt8](raw))
            f.write(encoded + " " + String(token_id) + "\n")
```

### 5b. `load_tiktoken(path: String)`

```mojo
def load_tiktoken(mut self, path: String) raises:
    var file_content = String()
    with open(path, "r") as f:
        file_content = f.read()
    var lines = file_content.split("\n")

    var mergeable_ranks = Dict[String, Int]()
    var all_tokens = List[List[UInt8]]()
    var max_id = 0

    for line_ptr in lines:
        var line = String(line_ptr.strip())
        if line.byte_length() == 0:
            continue
        var parts = line.split(" ")
        var raw = b64decode(parts[0])
        var rank = Int(parts[1])
        var key = _bytes_key(Span[UInt8](raw))
        mergeable_ranks[key] = rank
        while len(all_tokens) <= rank:
            all_tokens.append(List[UInt8]())
        all_tokens[rank] = raw^
        if rank > max_id:
            max_id = rank

    var new_vocab_size = max_id + 1

    # Build vocab display strings (byte_to_cp mapping)
    var new_vocab = List[String](capacity=new_vocab_size)
    var new_token_bytes = List[UInt8]()
    var new_token_offsets = List[Int](capacity=new_vocab_size + 1)
    var new_token_lengths = List[Int](capacity=new_vocab_size)
    for token_id in range(new_vocab_size):
        var raw_bytes = Span[UInt8](all_tokens[token_id])
        var display = String(capacity=len(raw_bytes))
        for i in range(len(raw_bytes)):
            display += chr(self.byte_to_cp[Int(raw_bytes[i])])
        new_vocab.append(display)
        # Build decode storage with Ġ→0x20 substitution
        new_token_offsets.append(len(new_token_bytes))
        var pending: Int = -1
        for cp in display.codepoints():
            var b = self.cp_to_byte[Int(cp)]
            if b == 0xA0 and pending == 0xC4:
                new_token_bytes.append(UInt8(0x20))
                pending = -1
            else:
                if pending >= 0:
                    new_token_bytes.append(UInt8(pending))
                pending = b
        if pending >= 0:
            new_token_bytes.append(UInt8(pending))
        new_token_lengths.append(
            len(new_token_bytes) - new_token_offsets[len(new_token_offsets) - 1]
        )
    new_token_offsets.append(len(new_token_bytes))

    # Recover merges
    var recovered = _recover_merges(mergeable_ranks, all_tokens)

    # Rebuild merge cache
    var new_merge_cache = PairCache()
    for merge in recovered:
        new_merge_cache.set(merge.first, merge.second, merge.merged)

    # Atomically swap state
    self.vocab_size = new_vocab_size
    self.vocab = new_vocab^
    self.token_bytes = new_token_bytes^
    self.token_offsets = new_token_offsets^
    self.token_lengths = new_token_lengths^
    self.merges = recovered^
    self.merge_cache = new_merge_cache^
```

---

## 6. Mojo stdlib APIs Used

| API | Signature | Use |
|---|---|---|
| `from std.base64 import b64encode` | `b64encode(input: Span[UInt8]) -> String` | Encode token bytes |
| `from std.base64 import b64decode` | `b64decode(str: StringSlice) -> List[UInt8]` | Decode base64 lines |
| `String(from_utf8=...)` | `String(*, from_utf8: Span[UInt8])` | (not needed for load — see §4a) |
| `chr` | `chr(cp: Int) -> String` | Safe codepoint → display char |

`b64encode` / `b64decode` are from Mojo stdlib 1.0.0b2 — no external module needed.
The `chr` function converts an Int codepoint to a single-character String.

---

## 7. File Changes (as-implemented)

```
mbpe/
├── bpe/tokenizer.mojo       # _bytes_key, _bpe, _recover_merges,
│                            # save_tiktoken, load_tiktoken
├── main.mojo                # test_save_tiktoken, test_tiktoken_roundtrip,
│                            # test_load_o200k_base
```

Implemented within existing files. No external dependencies.

---

## 8. Edge Cases

| Case | Handling |
|---|---|
| Empty file | No lines parsed → `vocab_size = 0`, no merges |
| Missing single-byte entries | Rare but possible.  Self test ensures `all_tokens[b]` exists for 0..255 |
| Non-base64 content in line | `b64decode` raises `Error` |
| `len(parts) < 2` on line | Malformed — skip (defensive) |
| Merge recovery fails for a token | Token bytes still stored in `vocab` for decode; merge simply omitted.  Encoding may produce suboptimal tokens but won't crash. |
| Duplicate rank | Last occurrence wins (dict overwrite) |
| Vocab with gaps (non-contiguous ranks) | `all_tokens` padded with empty `List[UInt8]()` up to `max_id`.  Gaps in merge range silently skipped. |
| Ġ→0x20 false positive | Two tokens whose `cp_to_byte` values happen to be 0xC4 then 0xA0.  Extremely rare (0xA0 is NBSP, non-printable).  Decode produces space instead — indistinguishable in most text. |
| Raw bytes not valid UTF-8 | .tiktoken stores raw bytes, which may not be valid UTF-8 (e.g., `[0x00]`).  We never call `String(from_utf8=raw_bytes)` — always go through `byte_to_cp` + `chr` for display. |

---

## 9. Implementation Order

| Step | File | What |
|---|---|---|
| 1 | `tokenizer.mojo` | Add `_bytes_key` static method |
| 2 | `tokenizer.mojo` | Add `_bpe` static method |
| 3 | `tokenizer.mojo` | Add `_recover_merges` method |
| 4 | `tokenizer.mojo` | Add `save_tiktoken` method |
| 5 | `tokenizer.mojo` | Add `load_tiktoken` method |
| 6 | `main.mojo` | Add `test_save_tiktoken` — train → save → verify lines parse |
| 7 | `main.mojo` | Add `test_tiktoken_roundtrip` — train → save → load → encode == original encode |
| 8 | `main.mojo` | Add `test_load_o200k_base` — load `bpe.mojo/data/o200k_base.tiktoken`, verify roundtrip |
| 9 | Benchmark | Verify performance parity after save → load → encode roundtrip |

---

## 10. Performance Choke Points

### 10a. Choke point: `_bytes_key` in the `_bpe` inner loop

**What:** For each adjacent-pair check in `_bpe`, `_bytes_key` builds a
comma-separated String (e.g. `"97,98,99"`) for Dict lookup.  This is called
on EVERY pair at EVERY merge iteration.

**Mitigation:** Pre-allocate with `capacity=len(bytes) * 4` (3 digits max per
byte + 1 comma).  Mojo `String(capacity=N)` allocates once; subsequent `+=`
operators write into the existing buffer without reallocation.

```mojo
fn _bytes_key(bytes: Span[UInt8]) -> String:
    var key = String(capacity=len(bytes) * 4)
    for i in range(len(bytes)):
        if i > 0:
            key += ","
        key += String(Int(bytes[i]))
    return key^
```

**Not optimized:** Dict lookup + hash computation on every pair check.  The
alternative — caching keys for unchanged pairs between merge iterations —
adds complexity with unclear benefit.  `_bpe` is called only during
`load_tiktoken`, not during encode/decode.

### 10b. Choke point: `List[UInt8]` concatenation in `_bpe`

**What:** Every pair check creates a temporary `List[UInt8]` (concatenation of
parts[i] + parts[i+1]).  Every merge creates another `List[UInt8]` + a new
`parts` list.  This is O(p²) small allocations per token where p = number of
byte-level parts.

**Mitigation:** Pre-allocate all temporaries with exact capacity:

| Allocation | Pre-alloc capacity |
|---|---|
| Initial `parts` | `len(token_bytes)` |
| Each `concat` | `len(parts[i]) + len(parts[i+1])` |
| `new_parts` per merge | `len(parts) - 1` |
| `merged` | `len(parts[i]) + len(parts[i+1])` |

### 10c. Choke point: Display string building in `load_tiktoken`

**What:** For each token, `display += chr(byte_to_cp[b])` appends 1–3 UTF-8
bytes per codepoint.  Unknown total UTF-8 length.

**Mitigation:** Pre-allocate `display = String(capacity=len(raw_bytes) * 3)`.
Worst case: all bytes are non-printable (masked to codepoints 256–511),
each encoding to 2 UTF-8 bytes → `len(raw_bytes) * 2`.  Using `* 3` gives
headroom for the null byte case (`\0` → codepoint 256 → 2 UTF-8 bytes).

### 10d. Non-issue: `save_tiktoken` reconstruction

Reconstructing raw bytes from display codepoints via `cp_to_byte` is O(total
vocab bytes).  Save is called once, not on any hot path.  Not worth
storing a separate `token_raw_bytes` field.

### 10e. `as_bytes()` usage summary

`vocab[id].as_bytes()` returns `Span[UInt8]` — a **zero-copy borrowed view**.
It is NOT used in `save_tiktoken` because display-string UTF-8 bytes differ
from raw BPE token bytes.  It IS used implicitly via `StringSlice` operations
in `load_tiktoken` (e.g. `b64decode()` takes `StringSlice`, which is what
`.split()` returns — no copy).

| Context | Uses `as_bytes()`? | Why |
|---|---|---|
| `save_tiktoken` | No | Need cp_to_byte mapping, not UTF-8 display bytes |
| `load_tiktoken` parse | No | `.split()` returns `StringSlice` (borrowed view of file content) |
| `load_tiktoken` display build | No | Iterating raw bytes from `all_tokens[rank]` |
| `_bytes_key` | Yes | Takes `Span[UInt8]`, callers pass `Span[UInt8](raw)` |
| `_bpe` | Yes | Takes `Span[UInt8]` for the token bytes parameter |

