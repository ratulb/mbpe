# Byte Mapping Refactor — Design & Implementation Plan

## 1. Problem Statement

Currently, `BPETokenizer` hardcodes the GPT-2 byte→ID mapping everywhere:

- **Encode hot path** (`_tokenize` line 468): `dst[i] = Int(ptr[i])` — assumes rank = byte value
- **Train** (`train` line 320): `ids.append(Int(sb[i]))` — same assumption
- **Load .tiktoken** (`load_tiktoken` line 794): `chr(self.byte_to_cp[Int(raw_bytes[i])])` — always uses GPT-2's `byte_to_cp`
- **Save .tiktoken** (`save_tiktoken` line 756): `self.cp_to_byte[Int(cp)]` — same

This works for r50k_base and cl100k_base (where rank 0 = byte 0x00), but breaks for
**o200k_base** (GPT-4o), where rank 0 = byte 0x21 (`!`), rank 220 = byte 0x20 (space), etc.

### o200k_base byte permutation

| Byte range | Ranks | Pattern |
|---|---|---|
| 0x21–0x7E (printable ASCII) | 0–93 | `rank = byte - 0x21` |
| 0xA1–0xAC, 0xAE–0xFF (extended) | 94–187 | contiguous block |
| 0x00–0x1F (control) | 188–219 | contiguous block |
| 0x20 (space) | 220 | single |
| 0x7F–0xA0 (DEL + gap) | 221–254 | contiguous block |
| 0xAD (soft hyphen) | 255 | single |

**Goal:** Make the pre-tokenizer the single source of truth for byte mapping, enabling
o200k_base support with zero overhead for the sequential (r50k/cl100k) case.

---

## 2. Design Principles

1. **Pre-tokenizer is the authority** — it defines split regex, byte mapping, and display conventions
2. **Comptime resolution** — byte mapping branches eliminated at compile time
3. **Zero overhead** — SEQUENTIAL (identity) mapping compiles to nothing
4. **Extensible** — new byte permutations add a new enum value + LUT, no core changes
5. **Backward compatible** — all existing tests pass unchanged throughout

---

## 3. Architecture Overview

```
PreTokenizer trait
├── comptime byte_map: ByteMapping     ← NEW: which byte→ID mapping
├── byte_to_id(b) -> Int               ← NEW: raw byte → base token rank
├── id_to_byte(rank) -> Int            ← NEW: base token rank → raw byte
├── split(text) -> List[String]        ← existing: regex-based word splitting
│
├── Shared whitespace matchers          ← NEW: default static methods on trait
│   ├── match_trailing_all_ws()
│   ├── match_ws_not_before_nonws()
│   └── match_single_ws()
│
├── GPreTokenizer                      ← unchanged (replace-based split)
├── GPT2Pretokenizer                   ← uses shared matchers + 4 family-specific
└── GPT4Pretokenizer[mapping]          ← uses shared matchers + 5 family-specific
    ├── [ByteMapping.SEQUENTIAL]       ← cl100k_base (rank = byte)
    └── [ByteMapping.SHUFFLED]         ← o200k_base (permuted ranks)

BPETokenizer[PT: PreTokenizer]
├── byte_to_cp / cp_to_byte           ← STAYS: universal GPT-2 bytes_to_unicode
├── train()                            ← UPDATED: PT.byte_to_id()
├── _tokenize()                        ← UPDATED: comptime PT.byte_map branch
├── load_tiktoken()                    ← UPDATED: PT.byte_to_id() for display
└── save_tiktoken()                    ← UPDATED: PT.id_to_byte() for raw bytes
```

---

## 4. Components

### 4.1 ByteMapping Enum

```mojo
@fieldwise_init
struct ByteMapping(ImplicitlyCopyable & Equatable):
    var _value: Int
    comptime SEQUENTIAL = ByteMapping(0)  # rank = byte value (r50k, cl100k)
    comptime SHUFFLED  = ByteMapping(1)  # o200k permutation
```

A simple integer-backed enum. Comptime values enable `comptime if` branching.
Extensible: add `ByteMapping(2)` for future permutations.

### 4.2 PreTokenizer Trait Changes

```mojo
trait PreTokenizer(Movable & Defaultable & ImplicitlyDeletable):
    # ── NEW: byte mapping ──────────────────────────────────
    comptime byte_map: ByteMapping  # required — each PT sets this

    @staticmethod
    fn byte_to_id(b: Int) -> Int:
        return b  # default: identity (sequential)

    @staticmethod
    fn id_to_byte(rank: Int) -> Int:
        return rank  # default: identity (sequential)

    # ── NEW: shared whitespace matchers (default static methods) ──
    @staticmethod
    fn match_trailing_all_ws(span: Span[UInt8], pos: Int) -> Int:
        """Match if ALL remaining bytes from pos to EOF are ASCII whitespace.
        Returns total remaining byte count, or 0 if any non-ws found."""
        ...  # implementation below

    @staticmethod
    fn match_ws_not_before_nonws(span: Span[UInt8], pos: Int) -> Int:
        """Match a whitespace run where the byte AFTER the run is also
        whitespace (or we're at EOF). Shrinks from right. Returns match length."""
        ...  # implementation below

    @staticmethod
    fn match_single_ws(span: Span[UInt8], pos: Int) -> Int:
        """Match a single ASCII whitespace byte. Returns 1 or 0."""
        ...  # implementation below

    # ── EXISTING: word splitting ───────────────────────────
    def split(self, text: String) raises -> List[String]:
        ...
```

**Why static methods on the trait (not free functions)?**

- **Discoverable** — all PT capabilities are in one place (the trait)
- **Overridable** — a future PT can override a shared matcher if needed
- **Organized** — matches the existing pattern (trait has `split`, now also has matchers)
- **Mojo supports it** — traits can have default static method implementations

**Mojo docs confirmation:**
> Traits can require static methods. A trait can supply a default implementation,
> so conforming structs don't need to implement the method themselves.

### 4.3 Shared Matcher Implementations

The three shared matchers are functionally identical between GPT2 and GPT4.
We extract one canonical implementation as a trait default.

**`match_trailing_all_ws`:**
```mojo
@staticmethod
fn match_trailing_all_ws(span: Span[UInt8], pos: Int) -> Int:
    # All remaining bytes must be ASCII whitespace.
    # Returns count of remaining bytes, or 0 if any non-ws found.
    var n = len(span)
    var i = pos
    while i < n:
        if not is_ascii_ws_byte(Int(span[i])):
            return 0
        i += 1
    return n - pos
```

**`match_ws_not_before_nonws`:**
```mojo
@staticmethod
fn match_ws_not_before_nonws(span: Span[UInt8], pos: Int) -> Int:
    # Whitespace run where the byte AFTER is also whitespace (or EOS).
    # Shrink-from-right: consume ws, then check if next byte is non-ws.
    var n = len(span)
    if not is_ascii_ws_byte(Int(span[pos])):
        return 0
    var end = pos + 1
    while end < n and is_ascii_ws_byte(Int(span[end])):
        end += 1
    # end is now past the ws run. If next byte is non-ws, shrink back.
    if end < n and not is_ascii_ws_byte(Int(span[end])):
        end -= 1
        if end == pos:
            return 0  # single ws followed by non-ws — no match
    return end - pos
```

**`match_single_ws`:**
```mojo
@staticmethod
fn match_single_ws(span: Span[UInt8], pos: Int) -> Int:
    if is_ascii_ws_byte(Int(span[pos])):
        return 1
    return 0
```

**Helper (module-level, already exists):**
```mojo
fn is_ascii_ws_byte(b: Int) -> Bool:
    return b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0B or b == 0x0C or b == 0x0D
```

### 4.4 How Each PT Uses Shared Matchers

**GPreTokenizer** — no change (replace-based split, no matchers).

**GPT2Pretokenizer:**
```
Before: 7 methods (4 family + 3 shared) + _best_match + split = 316 lines
After:  4 family methods + inherited shared + _best_match + split = ~240 lines
```
- Removes: `match_trailing_ws`, `match_ws_not_before_nonws`, `match_single_ws` (trait defaults)
- Updates `_best_match` to call `Self.match_trailing_all_ws(...)` etc.
- Keeps: `_match_contraction`, `_match_letter_run`, `_match_digit_run`, `_match_punct_run` (private struct methods)

**GPT4Pretokenizer:**
```
Before: 10 methods (5 family + 3 shared + 2 GPT4-only) + _best_match + split = 416 lines
After:  6 family methods + inherited shared + _best_match + split = ~330 lines
```
- Removes: `match_trailing_all_ws`, `match_ws_not_before_nonws`, `match_single_ws` (trait defaults)
- Updates `_best_match` to call `Self.match_trailing_all_ws(...)` etc.
- Keeps: `_match_contraction`, `_match_letter_run`, `_match_digit_run`, `_match_punct_run`, `_match_newline` (private struct methods)
- Deletes: `_match_whitespace` (dead code — never called from dispatch)

### 4.5 GPT4Pretokenizer SHUFFLED Variant

```mojo
struct GPT4Pretokenizer[mapping: ByteMapping = ByteMapping.SEQUENTIAL](PreTokenizer):
    comptime byte_map: ByteMapping = Self.mapping

    @staticmethod
    fn byte_to_id(b: Int) -> Int:
        comptime if Self.mapping == ByteMapping.SHUFFLED:
            return O200K_BYTE_TO_ID[b]  # 256-entry comptime LUT
        else:
            return b

    @staticmethod
    fn id_to_byte(rank: Int) -> Int:
        comptime if Self.mapping == ByteMapping.SHUFFLED:
            return O200K_ID_TO_BYTE[rank]  # 256-entry comptime LUT
        else:
            return rank
```

**Usage:**
```mojo
# cl100k_base (GPT-4) — sequential, same as before
var tok_cl100k = BPETokenizer[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]()

# o200k_base (GPT-4o) — shuffled byte mapping
var tok_o200k = BPETokenizer[GPT4Pretokenizer[ByteMapping.SHUFFLED]]()
```

### 4.6 Comptime LUTs for o200k

Derived from `o200k_base.tiktoken` first 256 entries:

```mojo
# byte value → token rank (o200k_base)
comptime O200K_BYTE_TO_ID = List[Int](
    188, 189, 190, 191, 192, 193, 194, 195,  # 0x00-0x07
    196, 197, 198, 199, 200, 201, 202, 203,  # 0x08-0x0F
    204, 205, 206, 207, 208, 209, 210, 211,  # 0x10-0x17
    212, 213, 214, 215, 216, 217, 218, 219,  # 0x18-0x1F
    220,                                       # 0x20 (space)
    0,   1,  2,  3,  4,  5,  6,  7,          # 0x21-0x28
    8,   9, 10, 11, 12, 13, 14, 15,          # 0x29-0x30
    16, 17, 18, 19, 20, 21, 22, 23,          # 0x31-0x38
    24, 25, 26, 27, 28, 29, 30, 31,          # 0x39-0x40
    32, 33, 34, 35, 36, 37, 38, 39,          # 0x41-0x48
    40, 41, 42, 43, 44, 45, 46, 47,          # 0x49-0x50
    48, 49, 50, 51, 52, 53, 54, 55,          # 0x51-0x58
    56, 57, 58, 59, 60, 61, 62, 63,          # 0x59-0x60
    64, 65, 66, 67, 68, 69, 70, 71,          # 0x61-0x68
    72, 73, 74, 75, 76, 77, 78, 79,          # 0x69-0x70
    80, 81, 82, 83, 84, 85, 86, 87,          # 0x71-0x78
    88, 89, 90, 91, 92, 93,                  # 0x79-0x7E
    221,                                      # 0x7F (DEL)
    222, 223, 224, 225, 226, 227, 228, 229,  # 0x80-0x87
    230, 231, 232, 233, 234, 235, 236, 237,  # 0x88-0x8F
    238, 239, 240, 241, 242, 243, 244, 245,  # 0x90-0x97
    246, 247, 248, 249, 250, 251, 252, 253,  # 0x98-0x9F
    254,                                      # 0xA0
    94,  95,  96,  97,  98,  99, 100, 101,  # 0xA1-0xA8
    102, 103, 104, 105,                      # 0xA9-0xAC
    255,                                      # 0xAD
    106, 107, 108, 109, 110, 111, 112, 113,  # 0xAE-0xAF, 0xB0-0xB7
    114, 115, 116, 117, 118, 119, 120, 121,  # 0xB8-0xBF
    122, 123, 124, 125, 126, 127, 128, 129,  # 0xC0-0xC7
    130, 131, 132, 133, 134, 135, 136, 137,  # 0xC8-0xCF
    138, 139, 140, 141, 142, 143, 144, 145,  # 0xD0-0xD7
    146, 147, 148, 149, 150, 151, 152, 153,  # 0xD8-0xDF
    154, 155, 156, 157, 158, 159, 160, 161,  # 0xE0-0xE7
    162, 163, 164, 165, 166, 167, 168, 169,  # 0xE8-0xEF
    170, 171, 172, 173, 174, 175, 176, 177,  # 0xF0-0xF7
    178, 179, 180, 181, 182, 183, 184, 185,  # 0xF8-0xFF
    186, 187                                   # 0xFE-0xFF
)

# token rank → byte value (o200k_base) — inverse of above
comptime O200K_ID_TO_BYTE = List[Int](
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,  # rank 0-7
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,  # rank 8-15
    0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,  # rank 16-23
    0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40,  # rank 24-31
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,  # rank 32-39
    0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,  # rank 40-47
    0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,  # rank 48-55
    0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60,  # rank 56-63
    0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,  # rank 64-71
    0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70,  # rank 72-79
    0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,  # rank 80-87
    0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E,              # rank 88-93
    0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8,  # rank 94-101
    0xA9, 0xAA, 0xAB, 0xAC, 0xAE, 0xAF, 0xB0, 0xB1,  # rank 102-109
    0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9,  # rank 110-117
    0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF, 0xC0, 0xC1,  # rank 118-125
    0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,  # rank 126-133
    0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1,  # rank 134-141
    0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,  # rank 142-149
    0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0, 0xE1,  # rank 150-157
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,  # rank 158-165
    0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1,  # rank 166-173
    0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9,  # rank 174-181
    0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,              # rank 182-187
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,  # rank 188-195
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,  # rank 196-203
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,  # rank 204-211
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,  # rank 212-219
    0x20,                                               # rank 220
    0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86,  # rank 221-228
    0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E,  # rank 229-236
    0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96,  # rank 237-244
    0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E,  # rank 245-252
    0x9F, 0xA0, 0xAD                                   # rank 253-255
)
```

---

## 5. BPETokenizer Changes

### 5.1 What stays the same

- `byte_to_cp` / `cp_to_byte` dicts — universal GPT-2 bytes_to_unicode (same for all families)
- Ġ→0x20 substitution (lines 409, 607, 800) — correct for all families
- `decode()` — uses `token_bytes` directly, no byte↔rank conversion
- Merge loop, PairCache, vocab storage — untouched
- `save()` / `load()` — serialize `byte_to_cp` (display mapping), not byte→rank

### 5.2 What changes

**`train()` line 319-320:**
```mojo
# Before:
for i in range(len(sb)):
    ids.append(Int(sb[i]))

# After:
for i in range(len(sb)):
    ids.append(Self.PT.byte_to_id(Int(sb[i])))
```

**`_tokenize()` line 466-468:**
```mojo
# Before:
for i in range(n):
    dst[i] = Int(ptr[i])

# After:
for i in range(n):
    comptime if Self.PT.byte_map == ByteMapping.SHUFFLED:
        dst[i] = Self.PT.byte_to_id(Int(ptr[i]))  # LUT lookup
    else:
        dst[i] = Int(ptr[i])                       # zero-cost identity
```

### 5.3 What does NOT change (corrected)

**`load_tiktoken()` line 794 — NO change needed:**
```mojo
# This line stays as-is:
display += chr(self.byte_to_cp[Int(raw_bytes[i])])
```
`byte_to_cp` maps raw bytes → safe display codepoints (universal GPT-2
bytes_to_unicode). The raw bytes come from the .tiktoken file directly.
No rank mapping is involved here — we're building display strings from
raw bytes, not from ranks.

**`save_tiktoken()` line 756 — NO change needed:**
```mojo
# This line stays as-is:
raw.append(UInt8(self.cp_to_byte[Int(cp)]))
```
`cp_to_byte` maps display codepoints → raw bytes (universal inverse).
Again, no rank mapping involved — we're reconstructing raw bytes from
display codepoints.

**Key insight:** `byte_to_id`/`id_to_byte` are only needed when converting
between raw bytes and token RANKS — i.e., in the encode hot path and in
training. The tiktoken save/load paths work in the display-codepoint
domain, which uses the universal `byte_to_cp`/`cp_to_byte` dicts.

---

## 6. What Does NOT Change

| Component | Why unchanged |
|---|---|
| `byte_to_cp` / `cp_to_byte` dicts | Universal bytes_to_unicode — same for all families |
| Ġ→0x20 substitution (3 locations) | Display strings contain Ġ for all families when bytes_to_unicode maps space to U+0120 |
| `decode()` | Uses precomputed `token_bytes` — raw bytes, no rank conversion |
| `load_tiktoken()` display construction | Works in raw-byte domain: `byte_to_cp[raw_byte]` — no rank mapping needed |
| `save_tiktoken()` raw byte reconstruction | Works in codepoint domain: `cp_to_byte[cp]` — no rank mapping needed |
| Merge loop | Operates on token IDs — already correct |
| `PairCache` | Cache is ID-based — works with any mapping |
| `save()` / `load()` | Serializes display strings + byte_to_cp — universal |
| `MERGE_SHIFT` / `ENCODE_SHIFT` constants | Bit-packing is ID-based — works with any mapping |

---

## 7. Implementation Phases

### Phase 1: Extract shared regex matchers to trait static methods
**Files:** `pretokenizer.mojo`
**Risk:** Low — pure refactoring, no behavioral change

1. Add `match_trailing_all_ws`, `match_ws_not_before_nonws`, `match_single_ws`
   as default static methods on `PreTokenizer` trait
2. Update `GPT2Pretokenizer._best_match` to call `Self.match_*()` (trait defaults)
3. Remove the 3 duplicated methods from `GPT2Pretokenizer`
4. Update `GPT4Pretokenizer._best_match` to call `Self.match_*()` (trait defaults)
5. Remove the 3 duplicated methods from `GPT4Pretokenizer`
6. Delete `GPT4Pretokenizer._match_whitespace` (dead code)
7. Run tests — all 25 should pass unchanged

### Phase 2: Add ByteMapping enum + trait members
**Files:** `pretokenizer.mojo`
**Risk:** Low — adds new code, no existing behavior changed

1. Add `ByteMapping` enum (SEQUENTIAL=0, SHUFFLED=1)
2. Add `comptime byte_map: ByteMapping` (required) to `PreTokenizer` trait
3. Add `byte_to_id` / `id_to_byte` static methods with defaults to trait
4. Add `comptime byte_map = ByteMapping.SEQUENTIAL` to `GPreTokenizer`
5. Add `comptime byte_map = ByteMapping.SEQUENTIAL` to `GPT2Pretokenizer`
6. Add `comptime byte_map = ByteMapping.SEQUENTIAL` to `GPT4Pretokenizer`
7. Add `O200K_BYTE_TO_ID` / `O200K_ID_TO_BYTE` comptime LUTs
8. Add `mapping` comptime parameter to `GPT4Pretokenizer`
9. Add `byte_to_id` / `id_to_byte` overrides with comptime branch
10. Run tests — all 25 should pass unchanged

### Phase 3: Wire BPETokenizer to PT byte mapping
**Files:** `tokenizer.mojo`
**Risk:** Medium — changes hot path, but comptime branch preserves current behavior for SEQUENTIAL

1. Update `train()` line 320: `PT.byte_to_id(Int(sb[i]))`
2. Update `_tokenize()` line 467-468: comptime branch on `PT.byte_map`
3. Update `load_tiktoken()` line 794: `PT.byte_to_id(Int(raw_bytes[i]))`
4. Update `save_tiktoken()` line 756: `PT.id_to_byte(Int(cp))`
5. Run tests — all 25 should pass unchanged (all use SEQUENTIAL PT)

### Phase 4: Enable o200k encode/decode
**Files:** `main.mojo`, `tests/test_tokenizer.mojo`
**Risk:** Low — test changes only, new functionality

1. Update `test_load_o200k_base` to use `GPT4Pretokenizer[ByteMapping.SHUFFLED]`
2. Re-enable encode/decode roundtrip assertions
3. Update `test_tiktoken_load_parity` if needed
4. Add new test: `test_byte_mapping_shuffled` (encode "hello world" with o200k, verify IDs)
5. Add new test: `test_byte_mapping_roundtrip` (encode→decode with SHUFFLED, verify text recovery)
6. Run tests — should now be 27 tests, all passing

---

## 8. Test Strategy

### Existing tests (no changes needed)
All 25 existing tests use `BPETokenizer` with default PT (`GPreTokenizer`) or explicit
`GPT2Pretokenizer` / `GPT4Pretokenizer` — all SEQUENTIAL. They should pass unchanged
throughout all phases.

### New tests (Phase 4)
1. **`test_byte_mapping_shuffled`**: Load o200k_base, encode "hello world", verify
   first token IDs match Python tiktoken reference
2. **`test_byte_mapping_roundtrip`**: Encode text with SHUFFLED, decode, verify identical
3. **`test_byte_mapping_sequential_eq_identity`**: Verify `GPT4Pretokenizer[SEQUENTIAL].byte_to_id(b) == b`
   for all b in 0-255
4. **`test_byte_mapping_shuffled_permutation`**: Verify `GPT4Pretokenizer[SHUFFLED].byte_to_id(0x21) == 0`
   and `GPT4Pretokenizer[SHUFFLED].id_to_byte(0) == 0x21`

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Comptime trait defaults not supported | Low — docs confirm it | Required member (no default) as fallback |
| Comptime `List[Int]` constants not supported | Medium | Runtime `Dict` on PT struct as fallback |
| LUT indexing out of bounds | Low — 256 entries, b in 0-255 | Bounds check or assert |
| Performance regression in SEQUENTIAL path | Very low — comptime branch eliminates code | Benchmark before/after |
| Shared matcher extraction breaks edge cases | Low — pure refactoring | Existing 25 tests cover all matchers |

---

## 10. Future Extensibility

- **New byte permutation**: Add `ByteMapping(2)`, new LUT, new PT parameter value
- **Special tokens**: Independent of byte mapping — can be added separately
- **`byte_shuffle` (gap #4)**: Different from `ByteMapping` — shuffle is per-token, not per-vocab
- **Pre-trained weight loading**: Uses `load_tiktoken()` — now works with any byte mapping
