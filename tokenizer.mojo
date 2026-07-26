"""Byte Pair Encoding tokenizer — train, encode, decode, save, load.

Design philosophy
-----------------
The hot path (training merges, encoding) works exclusively with Int token IDs.
Strings are materialised only when the outside world needs them: building the
vocabulary display strings, decoding IDs back to readable text, and serialising
to/from JSON.  This keeps allocations off the critical loop.

Byte-level base vocabulary (GPT-2 style)
-----------------------------------------
Instead of scanning the training corpus for unique characters, we start with
all 256 byte values (0x00–0xFF) as the base vocabulary.  Every Unicode
codepoint decomposes into 1–4 UTF-8 bytes, so every possible input is
representable.  There is no UNK token — ID 0 is simply byte 0x00.

Since raw bytes 0–255 can't live in a String (many aren't valid UTF-8), GPT-2
introduced a `bytes_to_unicode` table: printable bytes map to themselves and
the remaining control/whitespace bytes map to unused Unicode codepoints ≥ 256.
This keeps BPE's string operations working on visible characters while
preserving every byte round-trip.

References
----------
- Sennrich, Haddow, Birch (2016) — "Neural Machine Translation of Rare Words
  with Subword Units"  https://arxiv.org/abs/1508.07909
- GPT-2 encoder.py  —  `bytes_to_unicode` table
  https://github.com/openai/gpt-2/blob/master/src/encoder.py
- Karpathy minBPE  —  clean educational Python implementations
  https://github.com/karpathy/minbpe
- Hugging Face NLP Course  —  Chapter 6 (Tokenizers)
  https://huggingface.co/learn/nlp-course/chapter6/5
"""

from std.pathlib import Path
from std.python import Python
from std.memory import alloc, memcpy
from std.atomic import Atomic, Ordering, fence
from std.sys import size_of
from std.base64 import b64encode, b64decode

from pretokenizer import PreTokenizer, GPreTokenizer


# ---------------------------------------------------------------------------
# MergeRule — a BPE merge: (a_id, b_id, merged_id)
#
# Replaces raw Tuple[Int, Int, Int] with named fields and standard traits.
# ---------------------------------------------------------------------------

struct MergeRule(ImplicitlyCopyable & Equatable):
    var first: Int
    var second: Int
    var merged: Int

    def __init__(out self, first: Int, second: Int, merged: Int):
        self.first = first
        self.second = second
        self.merged = merged

    def __init__(out self, *, copy: Self):
        self.first = copy.first
        self.second = copy.second
        self.merged = copy.merged

    def __init__(out self, *, deinit move: Self):
        self.first = move.first
        self.second = move.second
        self.merged = move.merged

    def __eq__(self, other: Self) -> Bool:
        return (
            self.first == other.first
            and self.second == other.second
            and self.merged == other.merged
        )

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def __hash__(self) -> Int:
        return (
            (self.first * 2654435761)
            ^ (self.second * 2246822519)
            ^ (self.merged * 3266489917)
        )

    def __str__(self) -> String:
        return (
            "("
            + String(self.first)
            + ", "
            + String(self.second)
            + ") → "
            + String(self.merged)
        )


# ---------------------------------------------------------------------------
# PairCache — O(1) (token-pair → merged-id) lookup table
#
# Two-tier design:
#   Fast path: flat Int array (1024 × 1024) for IDs < 1000.
#              Index = (a << 10) | b — one shift-or-and, one cache-line load.
#   Slow path: Dict[Int, Int] for IDs ≥ 1000, packed key = (a << 20) | b.
#
# Built once after training/loading, read-only during encoding.
# ---------------------------------------------------------------------------

comptime CACHE_SHIFT: Int = 10
comptime CACHE_SIZE: Int = 1000
comptime CACHE_ENTRIES: Int = 1 << (CACHE_SHIFT * 2)
comptime ENCODE_SHIFT: Int = 20
comptime ENCODE_MASK: Int = (1 << ENCODE_SHIFT) - 1
comptime SEP: Int = -1

struct PairCache(ImplicitlyCopyable & Movable):
    """Reference-counted two-tier merge-lookup cache.

    Layout (one contiguous allocation):
      [Atomic[UInt64] refcount (8 bytes)]  [Int × CACHE_ENTRIES data (8 MB)]
      ^-- _refcount                         ^-- _fast

    Copying bumps the refcount — the flat array is shared, not duplicated.
    The last drop frees the entire block.  _slow is always owned (deep-copied).
    """
    var _fast: UnsafePointer[Int, MutAnyOrigin]
    var _refcount: UnsafePointer[Atomic[DType.uint64], MutAnyOrigin]
    var _slow: Dict[Int, Int]

    comptime REFCOUNT_BYTES: Int = size_of[Atomic[DType.uint64]]()
    comptime ALLOC_BYTES: Int = size_of[Atomic[DType.uint64]]() + CACHE_ENTRIES * size_of[Int]()

    def __init__(out self):
        var alloc_ptr = alloc[UInt8](Self.ALLOC_BYTES)
        self._refcount = alloc_ptr.bitcast[Atomic[DType.uint64]]()
        self._refcount[] = Atomic[DType.uint64](1)
        self._fast = (alloc_ptr + Self.REFCOUNT_BYTES).bitcast[Int]()
        for i in range(CACHE_ENTRIES):
            self._fast[i] = -1
        self._slow = Dict[Int, Int]()

    def __init__(out self, *, copy: Self):
        """Implement Copyable trait."""
        self._fast = copy._fast
        self._refcount = copy._refcount
        _ = self._refcount[].fetch_add[ordering=Ordering.RELAXED](1)
        self._slow = copy._slow.copy()

    def __init__(out self, *, deinit move: Self):
        """Implement Movable trait."""
        self._fast = move._fast
        self._fast = move._fast
        self._refcount = move._refcount
        self._slow = move._slow^

    def __del__(deinit self):
        if self._refcount[].fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return
        fence[ordering=Ordering.ACQUIRE]()
        self._refcount.bitcast[UInt8]().free()

    @always_inline
    def set(mut self, id1: Int, id2: Int, merged_id: Int):
        if id1 < CACHE_SIZE and id2 < CACHE_SIZE:
            self._fast[(id1 << CACHE_SHIFT) | id2] = merged_id
        else:
            self._slow[(id1 << ENCODE_SHIFT) | id2] = merged_id

    @always_inline
    def get(self, id1: Int, id2: Int) -> Int:
        if id1 < CACHE_SIZE and id2 < CACHE_SIZE:
            return self._fast[(id1 << CACHE_SHIFT) | id2]
        return self._slow.get((id1 << ENCODE_SHIFT) | id2, -1)


# ---------------------------------------------------------------------------
# BPETokenizer
#
# States
# ------
#   vocab       : ID → display string         (List[String])
#   merges      : ordered merge rules         (List[MergeRule])
#   merge_cache : fast pair→merged-id lookup  (PairCache)
#   byte_to_cp  : raw byte → safe codepoint   (Dict[Int, Int])
#   cp_to_byte  : safe codepoint → raw byte   (Dict[Int, Int])
#
# Why an ordered list for merges?
# -------------------------------
# Dict iteration order is not guaranteed (hash-table).  If we stored merges
# in a dict, re-loading the same data could produce a different iteration
# order, which would apply merge rules in the wrong sequence and generate
# different token IDs for the same text.  A List preserves insertion order,
# so the merge sequence is deterministic across save/load cycles.
#
# Why keep `merges` when `merge_cache` already provides O(1) lookup?
# ------------------------------------------------------------------
#  1. Training record — `train()` appends (a_id, b_id, merged_id) in learn
#     order.  merged_id = len(vocab) at that point, which IS the rank.
#     Without `merges` we'd lose what was learned and in what order.
#  2. Compact serialisation — `save()` writes the merge list as a few KB.
#     Serialising the 8 MB PairCache flat array instead would bloat every
#     save file by 4000×.
#  3. Rebuild from truth — `load()` reconstructs `merge_cache` from
#     `merges` (a single linear scan).  The merge list is the source of
#     truth; the cache is a derived structure.
#
# Conclusion: `merges` is metadata/serialisation only — not on the encode
# hot path.  `merge_cache` is the structure that matters for throughput.
#
# Why Int IDs instead of strings in the hot loop?
# -----------------------------------------------
# Every allocation and comparison in the inner merge loop used to be on
# heap-allocated String objects (one per character).  By switching to Int
# token IDs the merge loop becomes:
#   a) integer comparison  (one register op instead of strncmp)
#   b) no per-character allocations during encoding
#   c) encode() is a simple passthrough — _tokenize returns IDs directly
# ---------------------------------------------------------------------------

struct BPETokenizer[PT: PreTokenizer = GPreTokenizer](Sized & Movable):
    var pt: Self.PT
    var vocab: List[String]
    var merges: List[MergeRule]
    var merge_cache: PairCache
    var byte_to_cp: Dict[Int, Int]
    var cp_to_byte: Dict[Int, Int]
    var token_bytes: List[UInt8]
    var token_offsets: List[Int]
    var token_lengths: List[Int]

    def __init__(out self):
        self.pt = Self.PT()
        self.vocab = List[String]()
        self.merges = List[MergeRule]()
        self.merge_cache = PairCache()
        self.byte_to_cp = Dict[Int, Int]()
        self.cp_to_byte = Dict[Int, Int]()
        self.token_bytes = List[UInt8]()
        self.token_offsets = List[Int]()
        self.token_lengths = List[Int]()
        # GPT-2 bytes_to_unicode mapping (fixed at init, used by all methods)
        var n = 0
        for b in range(256):
            var printable = False
            if b >= 0x21 and b <= 0x7E:
                printable = True
            elif b >= 0xA1 and b <= 0xAC:
                printable = True
            elif b >= 0xAE and b <= 0xFF:
                printable = True
            if printable:
                self.byte_to_cp[b] = b
                self.cp_to_byte[b] = b
            else:
                var cp = 256 + n
                self.byte_to_cp[b] = cp
                self.cp_to_byte[cp] = b
                n += 1

    # ── training ────────────────────────────────────────────────────────
    # The algorithm:
    #   1. Pre-tokenise the corpus and count word frequencies.
    #   2. Build the byte→safe-unicode mapping (GPT-2 style).
    #   3. Initialise the vocabulary: bytes 0–255 at IDs 0–255.
    #   4. Split every word into a list of base token IDs (one per byte).
    #   5. Repeatedly find the most frequent adjacent pair and merge it,
    #      appending the new token to the vocabulary.
    #
    # Step 5 is the core BPE loop.  Each merge:
    #   - records the pair (a_id, b_id) and the new merged_id in `merges`
    #   - replaces every occurrence of a_id followed by b_id with merged_id
    #   - appends the concatenated display string to `vocab`
    #
    # The loop stops when we reach vocab_size or when no pairs remain
    # (every word has been reduced to a single token).
    # ─────────────────────────────────────────────────────────────────────

    def train[mut: Bool, //, origin: Origin[mut=mut]](mut self, corpus: Span[String, origin], vocab_size: Int) raises:
        # ---- 1. Pre-tokenise and compute word frequencies ----------------
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = self.pt.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # ---- 2. Build byte ↔ safe-Unicode mapping -----------------------
        # (initialized in __init__ — nothing to do here)

        # ---- 3. Initialise vocabulary -----------------------------------
        # IDs 0–255 are the 256 byte values, each mapped to its safe-Unicode
        # representation.  Merge tokens are appended below.
        # token_bytes/token_offsates form a flat array: token i's bytes are
        # token_bytes[token_offsets[i]:token_offsets[i+1]].
        self.vocab = List[String](capacity=vocab_size)
        self.token_bytes = List[UInt8]()
        self.token_offsets = List[Int](capacity=vocab_size + 1)
        self.token_lengths = List[Int](capacity=vocab_size)
        for b in range(256):
            var display = chr(self.byte_to_cp[b])
            self.vocab.append(display)
            self.token_offsets.append(len(self.token_bytes))
            for cp in display.codepoints():
                self.token_bytes.append(UInt8(self.cp_to_byte[Int(cp)]))
            self.token_lengths.append(len(self.token_bytes) - self.token_offsets[b])

        # ---- 4. Build flat token-ID sequence with SEP sentinel ------------
        # Flatten the word-frequency structure into a single List[Int] with
        # SEP separators between words.  Each word appears `freq` times to
        # preserve frequency weighting.  Tokens are byte values 0-255.
        var ids = List[Int]()
        for item in word_freqs.items():
            var word = item.key
            var freq = item.value
            var sb = word.as_bytes()
            for _ in range(freq):
                if len(ids) > 0:
                    ids.append(SEP)
                for i in range(len(sb)):
                    ids.append(Int(sb[i]))

        # ---- 5. Count initial pair frequencies (one pass) -----------------
        # Pairs involving SEP are skipped (they can never be merged).
        var stats = Dict[Int, Int]()
        for i in range(len(ids) - 1):
            if ids[i] != SEP and ids[i + 1] != SEP:
                var key = (ids[i] << ENCODE_SHIFT) | ids[i + 1]
                stats[key] = stats.get(key, 0) + 1

        # ---- 6. Merge loop (incremental pair stats) -----------------------
        # Each iteration finds the most frequent pair, applies the merge via
        # a single scan of `ids`, and updates `stats` incrementally (only
        # the 5 pairs affected per occurrence are adjusted).
        self.merges = List[MergeRule]()
        while len(self.vocab) < vocab_size:
            # Find the most frequent pair that does not involve SEP.
            var best_pair: Tuple[Int, Int] = (0, 0)
            var max_freq = -1
            for item in stats.items():
                if item.value > max_freq:
                    var a = item.key >> ENCODE_SHIFT
                    var b = item.key & ENCODE_MASK
                    if a != SEP and b != SEP:
                        max_freq = item.value
                        best_pair = (a, b)
            if max_freq <= 0:
                break

            var a_id = best_pair[0]
            var b_id = best_pair[1]
            var merged_id = len(self.vocab)

            # Single scan: apply merge + update stats incrementally.
            var new_ids = List[Int](capacity=len(ids))
            var i = 0
            while i < len(ids):
                if ids[i] == SEP:
                    new_ids.append(SEP)
                    i += 1
                elif (
                    i < len(ids) - 1
                    and ids[i + 1] != SEP
                    and ids[i] == a_id
                    and ids[i + 1] == b_id
                ):
                    # Decrement destroyed pairs: (prev, a), (a, b), (b, next)
                    if len(new_ids) > 0 and new_ids[len(new_ids) - 1] != SEP:
                        var pk = (new_ids[len(new_ids) - 1] << ENCODE_SHIFT) | ids[i]
                        if pk in stats:
                            var nv = stats[pk] - 1
                            stats[pk] = nv if nv > 0 else 0
                    var mk = (ids[i] << ENCODE_SHIFT) | ids[i + 1]
                    if mk in stats:
                        var nv = stats[mk] - 1
                        stats[mk] = nv if nv > 0 else 0
                    if i + 2 < len(ids) and ids[i + 2] != SEP:
                        var nk = (ids[i + 1] << ENCODE_SHIFT) | ids[i + 2]
                        if nk in stats:
                            var nv = stats[nk] - 1
                            stats[nk] = nv if nv > 0 else 0

                    # Increment created pairs: (prev, new_id), (new_id, next)
                    if len(new_ids) > 0 and new_ids[len(new_ids) - 1] != SEP:
                        var pk2 = (new_ids[len(new_ids) - 1] << ENCODE_SHIFT) | merged_id
                        stats[pk2] = stats.get(pk2, 0) + 1
                    if i + 2 < len(ids) and ids[i + 2] != SEP:
                        var nk2 = (merged_id << ENCODE_SHIFT) | ids[i + 2]
                        stats[nk2] = stats.get(nk2, 0) + 1

                    new_ids.append(merged_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids^

            # Record the merge.
            self.merges.append(MergeRule(a_id, b_id, merged_id))
            self.merge_cache.set(a_id, b_id, merged_id)

            # Build display string and flat byte storage.
            var merged_str = self.vocab[a_id].copy() + self.vocab[b_id].copy()
            self.vocab.append(merged_str)
            self.token_offsets.append(len(self.token_bytes))
            var pending: Int = -1
            for cp in merged_str.codepoints():
                var b = self.cp_to_byte[Int(cp)]
                if b == 0xA0 and pending == 0xC4:
                    self.token_bytes.append(UInt8(0x20))
                    pending = -1
                else:
                    if pending >= 0:
                        self.token_bytes.append(UInt8(pending))
                    pending = b
            if pending >= 0:
                self.token_bytes.append(UInt8(pending))
            self.token_lengths.append(len(self.token_bytes) - self.token_offsets[len(self.token_offsets) - 1])
        # Sentinel: last offset equals total byte count.
        self.token_offsets.append(len(self.token_bytes))

    # ── encoding ─────────────────────────────────────────────────────────
    # The encoder uses greedy rank-based merge via PairCache:
    #   1. Pre-tokenise the input text into words.
    #   2. Split each word into its raw UTF-8 bytes → base token IDs.
    #   3. Repeatedly scan adjacent ID-pairs, find the lowest-rank
    #      (earliest-learned) merge, and apply it in-place.
    #   4. Stop when no mergeable pairs remain.
    #
    # The merge_cache (PairCache) provides O(1) pair→merged-id lookup,
    # eliminating dead-rule scans.  Each word requires at most N merge
    # passes where N is the word's token count (worst case: one merge
    # per pass).  Typical text does ~3–5 passes per word.
    #
    # Accepts StringSlice (any string view) so callers can pass any
    # string-like value without owning a String.
    # ─────────────────────────────────────────────────────────────────────

    def _tokenize[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> List[Int]:
        if text.byte_length() == 0:
            return List[Int]()
        # ---- 1. Pre-tokenise into words ---------------------------------
        var text_str = String(from_utf8=text.as_bytes())
        var words = self.pt.split(text_str)
        # ---- 2. Compute total bytes for a single allocation -------------
        var total_bytes = 0
        for word in words:
            total_bytes += word.byte_length()
        if total_bytes == 0:
            return List[Int]()

        # ---- 3. Single allocation + per-word merge ----------------------
        var result = List[Int]()
        result.resize(total_bytes, 0)
        var write_pos = 0

        for word in words:
            var ptr = word.unsafe_ptr()
            var n = word.byte_length()
            var dst = result.unsafe_ptr() + write_pos

            # Copy bytes as Ints
            for i in range(n):
                dst[i] = Int(ptr[i])

            # Greedy lowest-rank merge loop
            while n >= 2:
                var best_rank = -1
                var best_a = -1
                var best_b = -1
                var best_m = -1
                for i in range(n - 1):
                    var merged = self.merge_cache.get(dst[i], dst[i + 1])
                    if merged >= 0 and (best_rank < 0 or merged < best_rank):
                        best_rank = merged
                        best_a = dst[i]
                        best_b = dst[i + 1]
                        best_m = merged
                if best_rank < 0:
                    break
                n = _merge_inplace_ptr(dst, n, best_a, best_b, best_m)

            write_pos += n

        # Trim to actual used size
        result.resize(write_pos, 0)
        return result^

    def encode[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> List[Int]:
        return self._tokenize(text)

    # ── decoding ─────────────────────────────────────────────────────────
    # Decode uses precomputed raw bytes per token (token_bytes built during
    # train/load), avoiding per-codepoint Dict lookups and per-character
    # iteration on the hot path.  The pre-tokeniser's Ġ spacer (UTF-8 bytes
    # 0xC4 0xA0) is replaced with space (0x20) at the byte level, avoiding
    # the cost of a separate string-level replace.
    # ─────────────────────────────────────────────────────────────────────

    def decode[mut: Bool, //, origin: Origin[mut=mut]](self, ids: Span[Int, origin]) raises -> String:
        if len(ids) == 0:
            return String("")
        var total: Int = 0
        for id in ids:
            total += self.token_lengths[id]
        if total == 0:
            return String("")
        var buf = alloc[UInt8](total)
        var ptr = self.token_bytes.unsafe_ptr()
        var write_offset: Int = 0
        for id in ids:
            var n = self.token_lengths[id]
            if n > 0:
                memcpy(
                    dest=buf + write_offset,
                    src=ptr + self.token_offsets[id],
                    count=n,
                )
                write_offset += n
        # ---- 3. Interpret bytes as UTF-8 (lossy) ------------------------
        var result = String(from_utf8=Span[UInt8](ptr=buf, length=write_offset))
        buf.free()
        return result^

    def __len__(self) -> Int:
        return len(self.vocab)

    # ── serialization ────────────────────────────────────────────────────
    # We serialize through Python's json module for simplicity.  The format:
    #
    #   {
    #     "vocab": ["\x00", "!", "\"", ...],      # display strings
    #     "merges": [[1, 5, 256], ...],            # (a_id, b_id, merged_id)
    #     "byte_to_cp": [0, 256, 257, ...]         # 256-element lookup
    #   }
    #
    # cp_to_byte is not stored explicitly — it's reconstructed from
    # byte_to_cp on load (since it's the inverse mapping).
    #
    # Breaking change: old save files (character-level vocab) are not
    # compatible with this format.
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: String) raises:
        var json = Python.import_module("json")
        var data = Python.dict()

        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(token))
        data["vocab"] = py_vocab

        var py_merges = Python.list()
        for merge in self.merges:
            var entry = Python.list()
            entry.append(Python.int(merge.first))
            entry.append(Python.int(merge.second))
            entry.append(Python.int(merge.merged))
            py_merges.append(entry)
        data["merges"] = py_merges

        var py_byte_to_cp = Python.list()
        for b in range(256):
            py_byte_to_cp.append(Python.int(self.byte_to_cp[b]))
        data["byte_to_cp"] = py_byte_to_cp

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Reconstruct byte ↔ safe-codepoint mappings from the stored array.
        var py_byte_to_cp = data["byte_to_cp"]
        tok.byte_to_cp = Dict[Int, Int]()
        tok.cp_to_byte = Dict[Int, Int]()
        for b in range(256):
            var cp = Int(py=py_byte_to_cp[b])
            tok.byte_to_cp[b] = cp
            tok.cp_to_byte[cp] = b

        # Rebuild vocab (stoi is not needed — encoding uses byte arithmetic
        # and the merge list).
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        tok.token_bytes = List[UInt8]()
        tok.token_offsets = List[Int](capacity=len(py_vocab) + 1)
        tok.token_lengths = List[Int](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var display = String(py_vocab[i])
            tok.vocab.append(display)
            tok.token_offsets.append(len(tok.token_bytes))
            var pending: Int = -1
            for cp in display.codepoints():
                var b = tok.cp_to_byte[Int(cp)]
                if b == 0xA0 and pending == 0xC4:
                    tok.token_bytes.append(UInt8(0x20))
                    pending = -1
                else:
                    if pending >= 0:
                        tok.token_bytes.append(UInt8(pending))
                    pending = b
            if pending >= 0:
                tok.token_bytes.append(UInt8(pending))
            tok.token_lengths.append(len(tok.token_bytes) - tok.token_offsets[i])
        tok.token_offsets.append(len(tok.token_bytes))

        # Rebuild ordered merge list.
        var py_merges = data["merges"]
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            var a_id = Int(py=entry[0])
            var b_id = Int(py=entry[1])
            var merged_id = Int(py=entry[2])
            tok.merges.append(MergeRule(a_id, b_id, merged_id))
            tok.merge_cache.set(a_id, b_id, merged_id)

        return tok^

    # ── .tiktoken format support ──────────────────────────────────────

    @staticmethod
    def _bytes_key(bytes: Span[UInt8, _]) -> String:
        var key = String(capacity=len(bytes) * 4)
        for i in range(len(bytes)):
            if i > 0:
                key += ","
            key += String(Int(bytes[i]))
        return key^

    @staticmethod
    def _bpe(
        mergeable_ranks: Dict[String, Int],
        token_bytes: Span[UInt8, _],
        max_rank: Int,
    ) raises -> List[List[UInt8]]:
        var parts = List[List[UInt8]](capacity=len(token_bytes))
        for i in range(len(token_bytes)):
            var single = List[UInt8](capacity=1)
            single.append(token_bytes[i])
            parts.append(single^)

        while True:
            var min_idx = -1
            var min_rank = -1
            for i in range(len(parts) - 1):
                var concat = List[UInt8](
                    capacity=len(parts[i]) + len(parts[i + 1])
                )
                for j in range(len(parts[i])):
                    concat.append(parts[i][j])
                for j in range(len(parts[i + 1])):
                    concat.append(parts[i + 1][j])
                var key = BPETokenizer._bytes_key(Span[UInt8](concat))
                if key in mergeable_ranks:
                    var rank = mergeable_ranks[key]
                    if min_idx < 0 or rank < min_rank:
                        min_idx = i
                        min_rank = rank
            if min_idx < 0 or (max_rank >= 0 and min_rank >= max_rank):
                break

            var merged = List[UInt8](
                capacity=len(parts[min_idx]) + len(parts[min_idx + 1])
            )
            for j in range(len(parts[min_idx])):
                merged.append(parts[min_idx][j])
            for j in range(len(parts[min_idx + 1])):
                merged.append(parts[min_idx + 1][j])
            var new_parts = List[List[UInt8]](capacity=len(parts) - 1)
            for j in range(min_idx):
                new_parts.append(parts[j].copy())
            new_parts.append(merged^)
            for j in range(min_idx + 2, len(parts)):
                new_parts.append(parts[j].copy())
            parts = new_parts^
        return parts^

    def _recover_merges(
        mut self,
        mergeable_ranks: Dict[String, Int],
        all_tokens: List[List[UInt8]],
    ) raises:
        var size = len(all_tokens)
        var recovered = List[MergeRule]()
        for token_id in range(256, size):
            var token_bytes = all_tokens[token_id].copy()
            var n = len(token_bytes)
            if n <= 1:
                continue
            var parts = self._bpe(
                mergeable_ranks, Span[UInt8](token_bytes), token_id
            )
            var left_id = -1
            var right_id = -1

            if len(parts) == 2:
                var lk = BPETokenizer._bytes_key(Span[UInt8](parts[0]))
                var rk = BPETokenizer._bytes_key(Span[UInt8](parts[1]))
                if lk in mergeable_ranks and rk in mergeable_ranks:
                    left_id = mergeable_ranks[lk]
                    right_id = mergeable_ranks[rk]
            elif len(parts) > 2:
                var best_cr = -1
                for i in range(len(parts) - 1):
                    var concat = List[UInt8](
                        capacity=len(parts[i]) + len(parts[i + 1])
                    )
                    for k in range(len(parts[i])):
                        concat.append(parts[i][k])
                    for k in range(len(parts[i + 1])):
                        concat.append(parts[i + 1][k])
                    var ck = BPETokenizer._bytes_key(Span[UInt8](concat))
                    if ck in mergeable_ranks:
                        var cr = mergeable_ranks[ck]
                        if cr < token_id and cr > best_cr:
                            var lk2 = BPETokenizer._bytes_key(
                                Span[UInt8](parts[i])
                            )
                            var rk2 = BPETokenizer._bytes_key(
                                Span[UInt8](parts[i + 1])
                            )
                            if (
                                lk2 in mergeable_ranks
                                and rk2 in mergeable_ranks
                            ):
                                var lr = mergeable_ranks[lk2]
                                var rr = mergeable_ranks[rk2]
                                if lr < token_id and rr < token_id:
                                    best_cr = cr
                                    left_id = lr
                                    right_id = rr
            if left_id >= 0 and right_id >= 0:
                recovered.append(
                    MergeRule(left_id, right_id, token_id)
                )
        self.merges = recovered^

    def save_tiktoken(mut self, path: String) raises:
        with open(path, "w") as f:
            for token_id in range(len(self.vocab)):
                var raw = List[UInt8](capacity=4)
                var display = self.vocab[token_id]
                for cp in display.codepoints():
                    raw.append(UInt8(self.cp_to_byte[Int(cp)]))
                var encoded = b64encode(Span[UInt8](raw))
                f.write(encoded + " " + String(token_id) + "\n")

    def load_tiktoken(mut self, path: String) raises:
        var file_content: String
        with open(path, "r") as f:
            file_content = f.read()
        var raw_lines = file_content.split("\n")

        var mergeable_ranks = Dict[String, Int]()
        var all_tokens = List[List[UInt8]]()
        var max_id = 0

        for line_ptr in raw_lines:
            var line = String(line_ptr.strip())
            if line.byte_length() == 0:
                continue
            var parts = line.split(" ")
            var raw = b64decode(parts[0])
            var rank = Int(parts[1])
            var key = BPETokenizer._bytes_key(Span[UInt8](raw))
            mergeable_ranks[key] = rank
            while len(all_tokens) <= rank:
                all_tokens.append(List[UInt8]())
            all_tokens[rank] = raw^
            if rank > max_id:
                max_id = rank

        var new_vocab_size = max_id + 1
        var new_vocab = List[String](capacity=new_vocab_size)
        var new_token_bytes = List[UInt8]()
        var new_token_offsets = List[Int](capacity=new_vocab_size + 1)
        var new_token_lengths = List[Int](capacity=new_vocab_size)
        for token_id in range(new_vocab_size):
            var raw_bytes = Span[UInt8](all_tokens[token_id])
            var display = String(capacity=len(raw_bytes) * 3)
            for i in range(len(raw_bytes)):
                display += chr(self.byte_to_cp[Int(raw_bytes[i])])
            new_vocab.append(display)
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
                len(new_token_bytes)
                - new_token_offsets[len(new_token_offsets) - 1]
            )
        new_token_offsets.append(len(new_token_bytes))

        self._recover_merges(mergeable_ranks, all_tokens)

        var new_merge_cache = PairCache()
        for merge in self.merges:
            new_merge_cache.set(merge.first, merge.second, merge.merged)

        self.vocab = new_vocab^
        self.token_bytes = new_token_bytes^
        self.token_offsets = new_token_offsets^
        self.token_lengths = new_token_lengths^
        self.merge_cache = new_merge_cache^


# ═══════════════════════════════════════════════════════════════════════════
# Hot-path helpers — raw pointer operations, zero allocations
# ═══════════════════════════════════════════════════════════════════════════

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





