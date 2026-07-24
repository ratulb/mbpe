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
