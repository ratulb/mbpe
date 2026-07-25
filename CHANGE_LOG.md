# Change Log

All notable changes to this project are documented here.
Format: `YYYY-MM-DD` — brief header, body, motivation.

---

## 2026-07-23 — Byte-level base vocabulary

**Motivation:** Character-level base vocab only covered codepoints seen during
training; unseen characters (emojis, CJK, rare symbols) mapped to `<UNK>`. All
modern LLM tokenizers (GPT-2/4, Llama, Mistral) use a byte-level base — all 256
byte values — making the tokenizer lossless for any UTF-8 input.

**Changes:**

| Field | Before | After |
|---|---|---|
| base vocab | unique codepoints in corpus | bytes 0–255 |
| `<UNK>` | emitted for unknown codepoints | never emitted (all bytes known) |
| `char_to_id` | `Dict[Int, Int]` — codepoint→ID | removed (redundant) |
| `byte_to_cp` | — | `Dict[Int, Int]` — byte→safe Unicode |
| `cp_to_byte` | — | `Dict[Int, Int]` — safe Unicode→byte |
| encode split | `char_to_id.get(cp, 0)` | `Int(byte) + 1` (arithmetic, no lookup) |
| decode | join vocab strings directly | vocab → cp_to_byte → `from_utf8_lossy` → `Ġ`→space |
| save/load | vocab + merges | vocab + merges + `byte_to_cp` + `cp_to_byte` |

**Tradeoffs:**
- `bytes_to_unicode` table must be constructed (see GPT-2 `encoder.py`).
- Decode needs a reverse-mapping step before UTF-8 reconstruction.
- Old save files incompatible (need re-train).
- Merge rules now operate on "safe" Unicode strings rather than raw text —
  roundtrip is lossless.

**Files touched:**
- `tokenizer.mojo` — core refactor: `train`, `_tokenize`, `decode`, `save`, `load`
- `main.mojo` — no changes (public API unchanged)

---

## 2026-07-23 — Remove <UNK> token (byte-level makes it dead code)

**Motivation:** With byte-level base vocabulary, every valid UTF-8 input is
losslessly representable — every byte 0x00–0xFF has a token ID, and Mojo's
`String` is guaranteed valid UTF-8.  The `<UNK>` token (ID 0) was never
produced by `encode()` and could only be reached by explicitly decoding
`[0]` in a test.  Removing it simplifies the ID arithmetic and matches the
convention used by GPT-2/4, Llama 3, and Mistral (no UNK in byte-level
base).

**Changes:**

| Before | After |
|---|---|
| ID 0 = `<UNK>`, IDs 1–256 = bytes 0–255 | IDs 0–255 = bytes 0–255 directly |
| encode: `Int(byte) + 1` | encode: `Int(byte)` |
| `len(vocab)` after train | `257 + len(merges)` | `256 + len(merges)` |
| `decode([0])` returns `"<UNK>"` | `decode([0])` returns byte-0 token |

**Files touched:**
- `tokenizer.mojo` — remove `<UNK>` from vocab init, drop `+1` in byte→ID
- `main.mojo` — remove `decode([0])` UNK assertion

---

## 2026-07-25 — `_tokenize` hot-path redesign (planned)

**Motivation:** `_tokenize` is the encode hot path — every instruction matters. The
original design built a `List[List[Int]]` of all word splits upfront, then applied
merge rules via list-comprehension rebuilds (heap alloc per merge hit), then
flattened into the result. This design eliminates all per-word heap allocations,
bounds checks, and the separate byte-copy pass by folding the first merge rule
into the byte-to-Int copy and using raw pointer arithmetic throughout.

**Design:**

```
                  Span[UInt8]
                 (word.as_bytes())
                       |
                  unsafe_ptr()
                       |
            +----------+----------+
            | _merge_span_to_buf  |  <- first rule only
            | (UInt8* -> Int*,    |     reads bytes, writes Ints,
            |  no bounds checks)  |     merges if pair matches
            +----------+----------+
                       |
               result's buffer
               (UnsafePointer[Int])
                       |
            +----------+----------+
            | _merge_inplace_ptr   |  <- remaining rules
            | (Int* <-> Int*,      |     in-place write-pointer shift
            |  no bounds checks)   |
            +----------+----------+
                       |
              result[start : start+n]
         (trimmed via pop() -- one trim per word)
```

**Data structures:**

| Name | Type | Role |
|---|---|---|
| `result` | `List[Int]` | Output token sequence. Also serves as scratch buffer for the current word. |
| `ptr` | `UnsafePointer[UInt8, _]` | Read-only byte view via `word.as_bytes().unsafe_ptr()` — no bounds checks. |
| `dst` | `UnsafePointer[Int, MutAnyOrigin]` | Mutable write pointer into `result`'s buffer via `Span(result).unsafe_ptr() + start`. |
| `start` | `Int` | Offset in `result` where the current word's tokens begin. |
| `n` | `Int` | Current number of tokens for this word (shrinks with each merge rule). |

**Helper functions:**

```mojo
@always_inline
def _merge_span_to_buf(
    dst: UnsafePointer[Int, MutAnyOrigin],
    src: UnsafePointer[UInt8, _],
    n: Int,
    a: Int, b: Int, m: Int,
) -> Int:
    """
    Read bytes from src (UInt8*), write Ints to dst.
    If (a, b) matches adjacent bytes, write merged_id (m) instead.
    Returns new length after merges (always <= n).
    No bounds checks, zero allocations.
    """
    var w = 0
    var i = 0
    while i < n:
        if i < n - 1 and Int(src[i]) == a and Int(src[i + 1]) == b:
            dst[w] = m
            i += 2
        else:
            dst[w] = Int(src[i])
            i += 1
        w += 1
    return w


@always_inline
def _merge_inplace_ptr(
    buf: UnsafePointer[Int, MutAnyOrigin],
    n: Int,
    a: Int, b: Int, m: Int,
) -> Int:
    """
    In-place write-pointer shift on Int buffer.
    Scan with read pointer i, write to buf[w].
    Skips ahead by 2 when (a,b) matches, writes merged_id.
    Returns new length.
    """
    var w = 0
    var i = 0
    while i < n:
        if i < n - 1 and buf[i] == a and buf[i + 1] == b:
            buf[w] = m
            i += 2
        else:
            buf[w] = buf[i]
            i += 1
        w += 1
    return w
```

**`_tokenize` body:**

```mojo
def _tokenize[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](self, text: StringSlice[origin]) raises -> List[Int]:
    if text.byte_length() == 0:
        return List[Int]()

    var words = PreTokenizer.tokenize(text)
    var result = List[Int]()

    for word in words:
        var sb = word.as_bytes()
        var ptr = sb.unsafe_ptr()
        var n = len(sb)
        var start = len(result)

        # Reserve space in result (one bulk extension)
        for _ in range(n):
            result.append(0)
        var dst = Span(result).unsafe_ptr() + start

        # Byte copy + first merge rule in one pass
        var first = self.merges[0]
        n = _merge_span_to_buf(dst, ptr, n, first[0], first[1], first[2])

        # Remaining rules: in-place pointer merge
        for idx in range(1, len(self.merges)):
            var rule = self.merges[idx]
            n = _merge_inplace_ptr(dst, n, rule[0], rule[1], rule[2])

        # Trim excess
        while len(result) > start + n:
            _ = result.pop()

    return result^
```

**Gains vs original:**

| Concern | Original | After |
|---|---|---|
| Per-word `List[Int]` | 1 heap alloc + appends | 0 |
| `splits` list | 1 alloc for list-of-lists | removed |
| `result.extend(split.copy())` | 1 flat alloc + copy | removed |
| Merge rebuild | `[e for e in ...[:i]] + [m] + ...` (alloc per hit) | write-pointer shift (0 alloc) |
| Bounds check on byte read | `Span.__getitem__` | `unsafe_ptr()[i]` |
| Bounds check on merge read/write | `List.__getitem__` | raw `buf[i]` |
| Copy + first merge | 2 passes | 1 pass |

No algorithm change — same sequential rule application, same outputs.
