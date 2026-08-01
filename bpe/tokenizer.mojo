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
from std.memory import memcpy
from std.base64 import b64encode, b64decode
from std.os.env import getenv
from std.collections.binary_heap import BinaryHeap

from bpe.pretokenizer import (
    PreTokenizer,
    GPreTokenizer,
    GPT2Pretokenizer,
    GPT4Pretokenizer,
    ByteMapping,
    WordCounts,
)
from bpe.array import IntArray, ByteArray


# ---------------------------------------------------------------------------
# MergeRule — a BPE merge: (a_id, b_id, merged_id)
#
# Replaces raw Tuple[Int, Int, Int] with named fields and standard traits.
# ---------------------------------------------------------------------------


struct MergeRule(ImplicitlyCopyable & Equatable & Writable):
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

    def write_to[T: Writer](self, mut writer: T):
        writer.write(
            String("MergeRule(")
            + String(self.first)
            + String(", ")
            + String(self.second)
            + String(") → ")
            + String(self.merged)
            + String(")")
        )


# ---------------------------------------------------------------------------
# MergeLookup — O(1) (token-pair → merged-id) lookup table
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
# Heap candidates pack (rank, position) into one Int: rank << HEAP_SHIFT | idx.
# rank < 2^24 (vocab ≤ ~200K) and idx < 2^24 (word byte length); the key is
# negated when pushed because BinaryHeap pops the maximum.
comptime HEAP_SHIFT: Int = 24
comptime HEAP_MASK: Int = (1 << HEAP_SHIFT) - 1
# Words shorter than this are merged with the tight scan loop; longer words
# use the heap-driven merge (measured crossover on real corpora).
comptime SCAN_LIMIT: Int = 32


struct MergeLookup(ImplicitlyCopyable & Movable & Writable):
    """Two-tier merge-lookup cache.

    _fast is a flat List[Int] (1024 × 1024), index = (a << 10) | b;
    copies are deep (List has no refcounted sharing).  _slow is always
    owned (deep-copied).
    """

    var _fast: List[Int]
    var _slow: Dict[Int, Int]

    def __init__(out self):
        self._fast = List[Int](length=CACHE_ENTRIES, fill=-1)
        self._slow = Dict[Int, Int]()

    def __init__(out self, *, copy: Self):
        """Implement Copyable trait."""
        self._fast = List[Int](copy=copy._fast)
        self._slow = copy._slow.copy()

    def __init__(out self, *, deinit move: Self):
        """Implement Movable trait."""
        self._fast = move._fast^
        self._slow = move._slow^

    @always_inline
    def set(mut self, id1: Int, id2: Int, merged_id: Int):
        if id1 < CACHE_SIZE and id2 < CACHE_SIZE:
            self._fast.unsafe_ptr()[(id1 << CACHE_SHIFT) | id2] = merged_id
        else:
            self._slow[(id1 << ENCODE_SHIFT) | id2] = merged_id

    @always_inline
    def get(self, id1: Int, id2: Int) -> Int:
        if id1 < CACHE_SIZE and id2 < CACHE_SIZE:
            return self._fast.unsafe_ptr()[(id1 << CACHE_SHIFT) | id2]
        return self._slow.get((id1 << ENCODE_SHIFT) | id2, -1)

    def write_to[T: Writer](self, mut writer: T):
        writer.write(
            String("MergeLookup(capacity=") + String(CACHE_ENTRIES) + String(")")
        )


# ---------------------------------------------------------------------------
# TokenByteTable — flat per-token byte storage
#
# Layout:
#   bytes      : concatenated raw bytes of every token (flat allocation)
#   offsets    : start offset of each token in `bytes`
#   lengths    : byte length of each token
#   token i's bytes live at bytes[offsets[i] : offsets[i] + lengths[i]]
#
# Memory model:
#   - The byte pool is a flat List[Byte] (heap-backed, amortised growth),
#     exposing the unsafe_ptr() API the decode hot path uses.
#   - The index arrays are List[Int] for the same reason — raw-pointer
#     access via unsafe_ptr().
#   - Copies are deep (ImplicitlyCopyable); refcounting was dropped
#     because the index arrays cannot be shared cheaply.
#
# Invariants:
#   len(lengths) == len(vocab) in BPETokenizer
#   after finish(): len(offsets) == len(lengths) + 1, and
#   offsets[len(lengths)] == len(bytes)  (sentinel — read by
#   benchmarks/profile_decode.mojo via offsets[id + 1])
# ---------------------------------------------------------------------------


struct TokenByteTable(ImplicitlyCopyable & Movable & Sized & Writable):
    var bytes: List[Byte]
    var offsets: List[Int]
    var lengths: List[Int]

    def __init__(out self):
        self.bytes = List[Byte]()
        self.offsets = List[Int]()
        self.offsets.append(0)
        self.lengths = List[Int]()

    def __init__(out self, *, copy: Self):
        """Deep copy of the byte pool and both index arrays."""
        self.bytes = List[Byte](copy=copy.bytes)
        self.offsets = List[Int](copy=copy.offsets)
        self.lengths = List[Int](copy=copy.lengths)

    def __init__(out self, *, deinit move: Self):
        self.bytes = move.bytes^
        self.offsets = move.offsets^
        self.lengths = move.lengths^

    @always_inline
    def __len__(self) -> Int:
        return len(self.lengths)

    @always_inline
    def reserve(mut self, max_tokens: Int):
        self.offsets.reserve(max_tokens + 1)
        self.lengths.reserve(max_tokens)

    @always_inline
    def add(mut self, raw: Span[Byte, _]):
        """Append a token's raw bytes."""
        var new_size = len(self.lengths) + 1
        self.offsets.append(0)
        self.lengths.append(0)
        self.offsets[new_size - 1] = len(self.bytes)
        self.bytes.reserve(len(self.bytes) + len(raw))
        for i in range(len(raw)):
            self.bytes.append(raw[i])
        self.lengths[new_size - 1] = len(raw)
        self.offsets[new_size] = len(self.bytes)

    @always_inline
    def set_bytes(mut self, id: Int, raw: Span[Byte, _]):
        """Register a token at an exact id, padding gaps with empty tokens."""
        while len(self.lengths) <= id:
            self.offsets.append(len(self.bytes))
            self.lengths.append(0)
        if len(self.offsets) == len(self.lengths):
            self.offsets.append(len(self.bytes))
        self.offsets[id] = len(self.bytes)
        self.bytes.reserve(len(self.bytes) + len(raw))
        for i in range(len(raw)):
            self.bytes.append(raw[i])
        self.lengths[id] = len(raw)
        self.offsets[len(self.offsets) - 1] = len(self.bytes)

    @always_inline
    def finish(mut self):
        """Append the sentinel offset (== total bytes), if not already present."""
        if len(self.offsets) == len(self.lengths):
            self.offsets.append(len(self.bytes))


# ---------------------------------------------------------------------------
# BPETokenizer
#
# States
# ------
#   vocab       : ID → display string         (List[String])
#   merges      : ordered merge rules         (List[MergeRule])
#   lookup_table : fast pair→merged-id lookup  (MergeLookup)
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
# Why keep `merges` when `lookup_table` already provides O(1) lookup?
# ------------------------------------------------------------------
#  1. Training record — `train()` appends (a_id, b_id, merged_id) in learn
#     order.  merged_id = len(vocab) at that point, which IS the rank.
#     Without `merges` we'd lose what was learned and in what order.
#  2. Compact serialization — `save()` writes the merge list as a few KB.
#     Serializing the 8 MB MergeLookup flat array instead would bloat every
#     save file by 4000×.
#  3. Rebuild from truth — `load()` reconstructs `lookup_table` from
#     `merges` (a single linear scan).  The merge list is the source of
#     truth; the cache is a derived structure.
#
# Conclusion: `merges` is metadata/serialisation only — not on the encode
# hot path.  `lookup_table` is the structure that matters for throughput.
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

comptime Vocabulary = List[String]
comptime Byte = UInt8
comptime ByteSequence = List[Byte]

struct BPETokenizer[PT: PreTokenizer = GPreTokenizer](
    Sized & Movable & Writable
):
    var pt: Self.PT
    var vocab: Vocabulary
    var merges: List[MergeRule]
    var lookup_table: MergeLookup
    var byte_to_cp: Dict[Int, Int]
    var cp_to_byte: Dict[Int, Int]
    var byte_to_rank: IntArray
    var token_table: TokenByteTable
    var special_bytes: Dict[String, Int]
    var inverse_special: Dict[Int, String]

    def __init__(out self):
        self.pt = Self.PT()
        self.vocab = Vocabulary()
        self.merges = List[MergeRule]()
        self.lookup_table = MergeLookup()
        self.byte_to_cp = Dict[Int, Int]()
        self.cp_to_byte = Dict[Int, Int]()
        self.byte_to_rank = IntArray.with_capacity(256)
        for b in range(256):
            self.byte_to_rank.append(b)
        self.token_table = TokenByteTable()
        self.special_bytes = Dict[String, Int]()
        self.inverse_special = Dict[Int, String]()
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

    def register_special_tokens(mut self, tokens: Dict[String, Int]) raises:
        """Register special tokens that bypass BPE encoding.

        Special tokens are preserved as single IDs during encode()
        (no BPE splitting) and skipped in save_tiktoken().
        Each special token's display text IS its raw text
        (no bytes_to_unicode mapping applied).

        Args:
            tokens: Dict mapping special token text to its reserved ID.
        """
        for item in tokens.items():
            self._register_special_token(item.key, item.value)

    def _register_special_token(mut self, text: String, id: Int) raises:
        if text.byte_length() == 0:
            raise Error("special token text must not be empty")
        if text in self.special_bytes:
            raise Error("duplicate special token: " + text)
        while len(self.vocab) <= id:
            self.vocab.append(String())
        self.special_bytes[text] = id
        self.inverse_special[id] = text
        self.vocab[id] = text
        self.token_table.set_bytes(id, text.as_bytes())

    def _display_to_bytes(self, display: String) raises -> ByteSequence:
        """Convert a display string back to raw token bytes.

        Each codepoint maps through cp_to_byte.  The GPreTokenizer
        spacer (UTF-8 bytes 0xC4 0xA0) is collapsed to a single 0x20
        byte so decoded text shows a plain space instead of the spacer.
        """
        var raw = ByteSequence(capacity=display.byte_length())
        var pending: Int = -1
        for cp in display.codepoints():
            var b = self.cp_to_byte[Int(cp)]
            if b == 0xA0 and pending == 0xC4:
                raw.append(Byte(0x20))
                pending = -1
            else:
                if pending >= 0:
                    raw.append(Byte(pending))
                pending = b
        if pending >= 0:
            raw.append(Byte(pending))
        return raw^

    # ── training ────────────────────────────────────────────────────────
    # The algorithm:
    #   1. Pre-tokenise the corpus and count word frequencies.
    #   2. Build the byte→safe-unicode mapping (GPT-2 style).
    #   3. Initialise the vocabulary: bytes 0–255 at IDs 0–255.
    #   4. Split every word into a list of base token IDs (one per byte);
    #      each distinct word is stored ONCE with its frequency.
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
    #
    # Design B (see SYSTEM.md 4.1.9a): words live in a flat IntArray arena
    # (no SEP, no replication, one allocation per training run); the `where`
    # map tracks pair → affected word indices, so each merge scans only the
    # words that contain the pair, compacting them in place, and pair counts
    # are updated arithmetically with delta × word frequency instead of
    # rescanning the whole corpus.
    # ─────────────────────────────────────────────────────────────────────

    def train[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, corpus: Span[String, origin], vocab_size: Int) raises:
        if vocab_size < 256:
            raise Error(
                "vocab_size must be at least 256 to hold the base byte"
                " vocabulary"
            )
        # ---- 1. Pre-tokenise and compute word frequencies ----------------
        # Fused single pass: the matcher hands each word span (ptr, len)
        # straight to WordCounts, which hashes the bytes in place — no
        # String materialization, no per-word allocation, no Dict.  Order
        # is first-seen (the same guarantee as Dict's insertion order),
        # so the tie-break below is byte-for-byte unchanged.
        var word_counts = WordCounts()
        for text in corpus:
            self.pt.count_words(text, word_counts)

        # ---- 2. Build byte ↔ safe-Unicode mapping -----------------------
        # (initialized in __init__ — nothing to do here)

        # ---- 3. Initialise vocabulary -----------------------------------
        # IDs 0–255 are the 256 byte values, each mapped to its safe-Unicode
        # representation.  Merge tokens are appended below.
        # token_table.bytes is a flat array: token i's bytes are
        # bytes[offsets[i]:offsets[i]+lengths[i]].
        # For SEQUENTIAL: rank == byte, so vocab[rank] = display(byte).
        # For SHUFFLED:   rank != byte, so we use id_to_byte(rank) to find
        # the raw byte for each rank, ensuring vocab[rank] is correct.
        self.vocab = Vocabulary(capacity=vocab_size)
        self.token_table = TokenByteTable()
        self.token_table.reserve(vocab_size)
        self.byte_to_rank = IntArray.with_capacity(256)
        for b in range(256):
            self.byte_to_rank.append(b)
        for rank in range(256):
            var b = Self.PT.id_to_byte(rank)
            var display = chr(self.byte_to_cp[b])
            self.vocab.append(display)
            self.token_table.add(self._display_to_bytes(display))

        # ---- 4. Build per-word token-ID storage + initial pair counts -------
        # Design B (4.1.9a): each distinct word is stored ONCE in a flat
        # token arena (word_freqs iteration order) with its frequency in
        # word_freq; there is no SEP and no replication — frequency
        # weighting is applied arithmetically as delta × freq.  `where`
        # maps a pair key to the indices of the words containing it, so a
        # merge only touches affected words instead of rescanning the
        # corpus.  Insertion order of `stats` keys is word order,
        # left-to-right within a word — identical to the flat design, so
        # the best-pair tie-break is byte-for-byte reproducible.
        #
        # Storage: `arena` is one flat IntArray holding every word's token
        # IDs back-to-back; word i occupies arena[offs[i]:offs[i]+len[i]].
        # The merge loop compacts in place through a raw pointer into the
        # arena (no per-word List allocation — one alloc per training run).
        var total_tokens: Int = 0
        for ei in word_counts.order:
            total_tokens += word_counts.lengths[ei]
        var arena = IntArray.with_capacity(total_tokens)
        var word_offs = IntArray.with_capacity(word_counts.n_entries)
        var word_len = IntArray.with_capacity(word_counts.n_entries)
        var word_freq = IntArray.with_capacity(word_counts.n_entries)
        var stats = Dict[Int, Int]()
        var where = Dict[Int, List[Int]]()
        var wb = word_counts.bytes.unsafe_ptr()
        for ei in word_counts.order:
            var off = word_counts.offsets[ei]
            var ln = word_counts.lengths[ei]
            var freq = word_counts.counts[ei]
            var off_arena = len(arena)
            word_offs.append(off_arena)
            for i in range(ln):
                arena.append(Self.PT.byte_to_id(Int(wb[off + i])))
            word_len.append(ln)
            word_freq.append(freq)
            var iw = len(word_offs) - 1
            for i in range(off_arena, off_arena + ln - 1):
                var key = (
                    (arena[i] << ENCODE_SHIFT)
                    | arena[i + 1]
                )
                stats[key] = stats.get(key, 0) + freq
                if key not in where:
                    where[key] = List[Int]()
                where[key].append(iw)

        # ---- 5. Merge loop (per-word where_to_update) ----------------------
        # Each iteration finds the most frequent pair, then scans ONLY the
        # words known to contain it, compacting them in place with the same
        # i += 2 / last-emitted-token semantics as the flat design, and
        # adjusts `stats` incrementally (the 5 pairs affected per
        # occurrence, × word frequency).
        self.lookup_table = MergeLookup()
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
            var best_key = (a_id << ENCODE_SHIFT) | b_id
            var merged_id = len(self.vocab)

            # Snapshot the affected-word list; the same key's list is never
            # appended to during its own merge (created pairs always involve
            # the brand-new merged_id), so the snapshot length is stable.
            var snap = len(where[best_key])
            for wi in range(snap):
                var iw = where[best_key][wi]
                var freq = word_freq[iw]
                var start = word_offs[iw]
                var n = word_len[iw]
                var wt = arena.unsafe_ptr() + start
                var w = 0
                var i = 0
                while i < n:
                    if i < n - 1 and wt[i] == a_id and wt[i + 1] == b_id:
                        # Decrement destroyed pairs: (prev, a), (a, b),
                        # (b, next); prev is the last emitted token.
                        if w > 0:
                            var pk = (wt[w - 1] << ENCODE_SHIFT) | wt[i]
                            if pk in stats:
                                var nv = stats[pk] - freq
                                stats[pk] = nv if nv > 0 else 0
                        var mk = (wt[i] << ENCODE_SHIFT) | wt[i + 1]
                        if mk in stats:
                            var nv = stats[mk] - freq
                            stats[mk] = nv if nv > 0 else 0
                        if i + 2 < n:
                            var nk = (wt[i + 1] << ENCODE_SHIFT) | wt[i + 2]
                            if nk in stats:
                                var nv = stats[nk] - freq
                                stats[nk] = nv if nv > 0 else 0
                        # Increment created pairs: (prev, new_id),
                        # (new_id, next), and register the word.
                        if w > 0:
                            var pk2 = (wt[w - 1] << ENCODE_SHIFT) | merged_id
                            stats[pk2] = stats.get(pk2, 0) + freq
                            if pk2 not in where:
                                where[pk2] = List[Int]()
                            where[pk2].append(iw)
                        if i + 2 < n:
                            var nk2 = (merged_id << ENCODE_SHIFT) | wt[i + 2]
                            stats[nk2] = stats.get(nk2, 0) + freq
                            if nk2 not in where:
                                where[nk2] = List[Int]()
                            where[nk2].append(iw)
                        wt[w] = merged_id
                        w += 1
                        i += 2
                    else:
                        wt[w] = wt[i]
                        w += 1
                        i += 1
                word_len[iw] = w

            # Record the merge.
            self.merges.append(MergeRule(a_id, b_id, merged_id))
            self.lookup_table.set(a_id, b_id, merged_id)

            # Build display string and flat byte storage.
            var merged_str = self.vocab[a_id].copy() + self.vocab[b_id].copy()
            self.vocab.append(merged_str)
            self.token_table.add(self._display_to_bytes(merged_str))
        # Sentinel: last offset equals total byte count.
        self.token_table.finish()

    # ── encoding ─────────────────────────────────────────────────────────
    # The encoder uses greedy rank-based merge via MergeLookup:
    #   1. Pre-tokenise the input text into words.
    #   2. Split each word into its raw UTF-8 bytes → base token IDs.
    #   3. Repeatedly scan adjacent ID-pairs, find the lowest-rank
    #      (earliest-learned) merge, and apply it in-place.
    #   4. Stop when no mergeable pairs remain.
    #
    # The lookup_table (MergeLookup) provides O(1) pair→merged-id lookup,
    # eliminating dead-rule scans.  Each word requires at most N merge
    # passes where N is the word's token count (worst case: one merge
    # per pass).  Typical text does ~3–5 passes per word.
    #
    # Accepts StringSlice (any string view) so callers can pass any
    # string-like value without owning a String.
    # ─────────────────────────────────────────────────────────────────────

    def encode_ordinary[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> List[Int]:
        if text.byte_length() == 0:
            return List[Int]()
        # ---- 1. Pre-tokenise into words (zero-copy views) --------------
        ref words = self.pt.split_view(text)
        # ---- 2. Compute total bytes for a single allocation -------------
        var total_bytes = 0
        for word in words:
            total_bytes += word.byte_length()
        if total_bytes == 0:
            return List[Int]()

        # ---- 3. Single allocation + per-word heap-driven merge -----------
        # Each word is a linked list (prev/next arrays) over its token nodes.
        # A min-heap of candidates drives the greedy merge: pop the lowest-
        # rank pair, splice it in O(1), and re-push only the two pairs
        # touching the merge site.  O(N log N) instead of the O(N^2) full-
        # scan-per-merge of the original encoder.
        #
        # Heap keys pack (rank, position) into a single Int: rank << 24 | idx.
        # BinaryHeap is a max-heap, so the key is negated — the popped key
        # then yields the lowest rank, breaking ties by lowest position
        # (identical to the scan-based selection order).
        #
        # All scratch buffers are raw allocations reused across words; the
        # BinaryHeap retains its capacity between words, so steady-state
        # encoding does zero allocation.
        var result = List[Int]()
        result.resize(total_bytes, 0)
        var write_pos = 0
        var ids_buf = alloc[Int](0)
        var nxt_buf = alloc[Int](0)
        var prv_buf = alloc[Int](0)
        var alive_buf = alloc[UInt8](0)
        var cap = 0
        var heap = BinaryHeap[Int]()
        var btr = self.byte_to_rank.unsafe_ptr()
        for word in words:
            var ptr = word.unsafe_ptr()
            var n = word.byte_length()
            var dst = result.unsafe_ptr() + write_pos

            # Copy bytes as Ints (via PT byte mapping)
            if n < 2:
                for i in range(n):
                    comptime if Self.PT.byte_map == ByteMapping.SHUFFLED:
                        dst[i] = Self.PT.byte_to_id(Int(ptr[i]))
                    else:
                        dst[i] = btr[Int(ptr[i])]
                write_pos += n
                continue

            # Short words: tight scan-based greedy merge (O(n^2) is cheaper
            # than heap bookkeeping at this size).
            if n < SCAN_LIMIT:
                for i in range(n):
                    comptime if Self.PT.byte_map == ByteMapping.SHUFFLED:
                        dst[i] = Self.PT.byte_to_id(Int(ptr[i]))
                    else:
                        dst[i] = btr[Int(ptr[i])]
                var len = n
                while len >= 2:
                    var best_rank = -1
                    var best_a = -1
                    var best_b = -1
                    var best_m = -1
                    for i in range(len - 1):
                        var merged = self.lookup_table.get(dst[i], dst[i + 1])
                        if merged >= 0 and (best_rank < 0 or merged < best_rank):
                            best_rank = merged
                            best_a = dst[i]
                            best_b = dst[i + 1]
                            best_m = merged
                    if best_rank < 0:
                        break
                    len = merge_inplace(dst, len, best_a, best_b, best_m)
                write_pos += len
                continue

            # Long words: heap-driven merge (O(N log N)).
            if n > cap:
                ids_buf.free()
                nxt_buf.free()
                prv_buf.free()
                alive_buf.free()
                ids_buf = alloc[Int](n)
                nxt_buf = alloc[Int](n)
                prv_buf = alloc[Int](n)
                alive_buf = alloc[UInt8](n)
                cap = n

            for i in range(n):
                comptime if Self.PT.byte_map == ByteMapping.SHUFFLED:
                    ids_buf[i] = Self.PT.byte_to_id(Int(ptr[i]))
                else:
                    ids_buf[i] = btr[Int(ptr[i])]
                nxt_buf[i] = i + 1
                prv_buf[i] = i - 1
                alive_buf[i] = 1
            nxt_buf[n - 1] = -1

            # Seed the heap with every initially mergeable pair.
            for i in range(n - 1):
                var r0 = self.lookup_table.get(ids_buf[i], ids_buf[i + 1])
                if r0 >= 0:
                    heap.push(-(r0 << HEAP_SHIFT | i))

            # Greedy lowest-rank merge (lazy-validated heap).
            while len(heap) > 0:
                var key = -heap.pop()
                var e = key & HEAP_MASK
                var rank = key >> HEAP_SHIFT
                if alive_buf[e] == 0:
                    continue
                var j = nxt_buf[e]
                if j < 0:
                    continue
                # Revalidate: the pair may have changed since it was pushed.
                if self.lookup_table.get(ids_buf[e], ids_buf[j]) != rank:
                    continue

                # Merge: node e absorbs node j (e stays, j is spliced out).
                ids_buf[e] = rank
                var k = nxt_buf[j]
                if k >= 0:
                    prv_buf[k] = e
                nxt_buf[e] = k
                alive_buf[j] = 0

                # Only the two pairs touching the merge site can change.
                var p = prv_buf[e]
                if p >= 0:
                    var rp = self.lookup_table.get(ids_buf[p], ids_buf[e])
                    if rp >= 0:
                        heap.push(-(rp << HEAP_SHIFT | p))
                if k >= 0:
                    var rk = self.lookup_table.get(ids_buf[e], ids_buf[k])
                    if rk >= 0:
                        heap.push(-(rk << HEAP_SHIFT | e))

            # Emit surviving nodes in order (node 0 is never consumed).
            var count = 0
            var cur = 0
            while cur >= 0:
                dst[count] = ids_buf[cur]
                count += 1
                cur = nxt_buf[cur]
            write_pos += count

        ids_buf.free()
        nxt_buf.free()
        prv_buf.free()
        alive_buf.free()

        # Trim to actual used size
        result.resize(write_pos, 0)
        return result^

    def encode[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> List[Int]:
        if len(self.special_bytes) == 0:
            return self.encode_ordinary(text)

        var n = text.byte_length()
        if n == 0:
            return List[Int]()

        var bytes = text.as_bytes()
        var result = List[Int]()
        var pos = 0

        while pos < n:
            var found_id = -1
            var found_len = 0
            for item in self.special_bytes.items():
                var tok = item.key
                var tok_id = item.value
                var tok_len = tok.byte_length()
                if pos + tok_len <= n:
                    var matched = True
                    var tok_bytes = tok.as_bytes()
                    for k in range(tok_len):
                        if bytes[pos + k] != tok_bytes[k]:
                            matched = False
                            break
                    if matched:
                        found_id = tok_id
                        found_len = tok_len
                        break
            if found_id >= 0:
                result.append(found_id)
                pos += found_len
            else:
                var start = pos
                var next_special = n
                for item in self.special_bytes.items():
                    var tok = item.key
                    var found_at = text.find(tok, start)
                    if found_at >= 0 and found_at < next_special:
                        next_special = found_at
                if next_special > start:
                    var seg = StringSlice(unsafe_from_utf8=bytes[start:next_special])
                    for id in self.encode_ordinary(seg):
                        result.append(id)
                    pos = next_special
                elif next_special == start:
                    pos += 1
                else:
                    var seg = StringSlice(unsafe_from_utf8=bytes[start:n])
                    for id in self.encode_ordinary(seg):
                        result.append(id)
                    pos = n

        return result^

    # ── decoding ─────────────────────────────────────────────────────────
    # Decode uses precomputed raw bytes per token (token_table built during
    # train/load), avoiding per-codepoint Dict lookups and per-character
    # iteration on the hot path.  The pre-tokeniser's Ġ spacer (UTF-8 bytes
    # 0xC4 0xA0) is replaced with space (0x20) at the byte level, avoiding
    # the cost of a separate string-level replace.
    # ─────────────────────────────────────────────────────────────────────

    def decode[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, ids: Span[Int, origin]) raises -> String:
        if len(ids) == 0:
            return String("")
        var lens = self.token_table.lengths.unsafe_ptr()
        var offs = self.token_table.offsets.unsafe_ptr()
        var n_tokens = len(self.token_table)
        var total: Int = 0
        for id in ids:
            if id < 0 or id >= n_tokens:
                raise Error("token ID out of range: " + String(id))
            total += lens[id]
        if total == 0:
            return String("")
        var result = String(unsafe_uninit_length=total)
        var dst = result.as_bytes().unsafe_ptr().unsafe_mut_cast[True]()
        var ptr = self.token_table.bytes.unsafe_ptr().as_noalias_ptr()
        var write_offset: Int = 0
        for id in ids:
            var n = lens[id]
            if n > 0:
                memcpy(
                    dest=dst + write_offset,
                    src=ptr + offs[id],
                    count=n,
                )
                write_offset += n
        return result^

    def __len__(self) -> Int:
        return len(self.vocab)

    def name(self) -> String:
        return Self.PT.name()

    def decode_bytes[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, ids: Span[Int, origin]) raises -> ByteSequence:
        if len(ids) == 0:
            return ByteSequence()
        var lens = self.token_table.lengths.unsafe_ptr()
        var offs = self.token_table.offsets.unsafe_ptr()
        var n_tokens = len(self.token_table)
        var total: Int = 0
        for id in ids:
            if id < 0 or id >= n_tokens:
                raise Error("token ID out of range: " + String(id))
            total += lens[id]
        if total == 0:
            return ByteSequence()
        var result = ByteSequence(capacity=total)
        result.resize(total, 0)
        var dst = result.unsafe_ptr()
        var src = self.token_table.bytes.unsafe_ptr().as_noalias_ptr()
        var write_offset: Int = 0
        for id in ids:
            var n = lens[id]
            if n > 0:
                memcpy(dest=dst + write_offset, src=src + offs[id], count=n)
                write_offset += n
        return result^

    def decode_single_token_bytes(self, id: Int) raises -> ByteSequence:
        if id < 0 or id >= len(self.token_table):
            raise Error("token ID out of range: " + String(id))
        var n = self.token_table.lengths.unsafe_ptr()[id]
        if n == 0:
            return ByteSequence()
        var off = self.token_table.offsets.unsafe_ptr()[id]
        var ptr = self.token_table.bytes.unsafe_ptr().as_noalias_ptr()
        var result = ByteSequence(capacity=n)
        result.resize(n, 0)
        memcpy(dest=result.unsafe_ptr(), src=ptr + off, count=n)
        return result^

    def decode_with_offsets[
        mut: Bool, //, origin: Origin[mut=mut]
    ](
        self, ids: Span[Int, origin], mut starts: List[Int], mut ends: List[Int]
    ) raises -> String:
        if len(ids) == 0:
            return String("")
        var lens = self.token_table.lengths.unsafe_ptr()
        var offs = self.token_table.offsets.unsafe_ptr()
        var n_tokens = len(self.token_table)
        var total: Int = 0
        for id in ids:
            if id < 0 or id >= n_tokens:
                raise Error("token ID out of range: " + String(id))
            total += lens[id]
        if total == 0:
            return String("")
        var result = String(unsafe_uninit_length=total)
        var dst = result.as_bytes().unsafe_ptr().unsafe_mut_cast[True]()
        var ptr = self.token_table.bytes.unsafe_ptr().as_noalias_ptr()
        var write_offset: Int = 0
        for id in ids:
            var n = lens[id]
            starts.append(write_offset)
            if n > 0:
                memcpy(dest=dst + write_offset, src=ptr + offs[id], count=n)
                write_offset += n
            ends.append(write_offset)
        return result^

    def token_byte_values(self) -> List[ByteSequence]:
        var result = List[ByteSequence](capacity=len(self.token_table))
        var lens = self.token_table.lengths.unsafe_ptr()
        var offs = self.token_table.offsets.unsafe_ptr()
        var ptr = self.token_table.bytes.unsafe_ptr().as_noalias_ptr()
        for i in range(len(self.token_table)):
            var n = lens[i]
            var bytes = ByteSequence(capacity=n)
            bytes.resize(n, 0)
            memcpy(dest=bytes.unsafe_ptr(), src=ptr + offs[i], count=n)
            result.append(bytes^)
        return result^

    def encode_single_token[mut: Bool, //, origin: Origin[mut=mut]](self, text: StringSlice[origin]) raises -> Int:
        for item in self.special_bytes.items():
            if item.key == text:
                return item.value
        for i in range(len(self.vocab)):
            if self.vocab[i] == text:
                return i
        raise Error("unknown token: " + String(text))

    def write_to[T: Writer](self, mut writer: T):
        writer.write(
            String("BPETokenizer(vocab_size=")
            + String(len(self.vocab))
            + String(")")
        )

    # ── serialization ────────────────────────────────────────────────────
    # We use the standard .tiktoken format (OpenAI-compatible):
    #   <base64(token_bytes)> <rank>\n
    #
    # See save_tiktoken() / load_tiktoken() below.
    # The legacy JSON-based save()/load() has been removed.
    # ─────────────────────────────────────────────────────────────────────

    # ── .tiktoken format support ──────────────────────────────────────

    @staticmethod
    def _bytes_key(bytes: Span[Byte, _]) -> String:
        var key = String(capacity=len(bytes) * 4)
        for i in range(len(bytes)):
            if i > 0:
                key += ","
            key += String(Int(bytes[i]))
        return key^

    @staticmethod
    def _bpe(
        mergeable_ranks: Dict[String, Int],
        token_bytes: Span[Byte, _],
        max_rank: Int,
    ) raises -> List[ByteSequence]:
        var parts = List[ByteSequence](capacity=len(token_bytes))
        for i in range(len(token_bytes)):
            var single = ByteSequence(capacity=1)
            single.append(token_bytes[i])
            parts.append(single^)

        while True:
            var min_idx = -1
            var min_rank = -1
            for i in range(len(parts) - 1):
                var concat = ByteSequence(
                    capacity=len(parts[i]) + len(parts[i + 1])
                )
                for j in range(len(parts[i])):
                    concat.append(parts[i][j])
                for j in range(len(parts[i + 1])):
                    concat.append(parts[i + 1][j])
                var key = BPETokenizer._bytes_key(Span[Byte](concat))
                if key in mergeable_ranks:
                    var rank = mergeable_ranks[key]
                    if min_idx < 0 or rank < min_rank:
                        min_idx = i
                        min_rank = rank
            if min_idx < 0 or (max_rank >= 0 and min_rank >= max_rank):
                break

            var merged = ByteSequence(
                capacity=len(parts[min_idx]) + len(parts[min_idx + 1])
            )
            for j in range(len(parts[min_idx])):
                merged.append(parts[min_idx][j])
            for j in range(len(parts[min_idx + 1])):
                merged.append(parts[min_idx + 1][j])
            var new_parts = List[ByteSequence](capacity=len(parts) - 1)
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
        all_tokens: List[ByteSequence],
    ) raises:
        var size = len(all_tokens)
        var recovered = List[MergeRule]()
        for token_id in range(256, size):
            var token_bytes = all_tokens[token_id].copy()
            var n = len(token_bytes)
            if n <= 1:
                continue
            var parts = self._bpe(
                mergeable_ranks, Span[Byte](token_bytes), token_id
            )
            var left_id = -1
            var right_id = -1

            if len(parts) == 2:
                var lk = BPETokenizer._bytes_key(Span[Byte](parts[0]))
                var rk = BPETokenizer._bytes_key(Span[Byte](parts[1]))
                if lk in mergeable_ranks and rk in mergeable_ranks:
                    left_id = mergeable_ranks[lk]
                    right_id = mergeable_ranks[rk]
            elif len(parts) > 2:
                var best_cr = -1
                for i in range(len(parts) - 1):
                    var concat = ByteSequence(
                        capacity=len(parts[i]) + len(parts[i + 1])
                    )
                    for k in range(len(parts[i])):
                        concat.append(parts[i][k])
                    for k in range(len(parts[i + 1])):
                        concat.append(parts[i + 1][k])
                    var ck = BPETokenizer._bytes_key(Span[Byte](concat))
                    if ck in mergeable_ranks:
                        var cr = mergeable_ranks[ck]
                        if cr < token_id and cr > best_cr:
                            var lk2 = BPETokenizer._bytes_key(
                                Span[Byte](parts[i])
                            )
                            var rk2 = BPETokenizer._bytes_key(
                                Span[Byte](parts[i + 1])
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
                recovered.append(MergeRule(left_id, right_id, token_id))
        self.merges = recovered^

    def save_tiktoken(mut self, path: String) raises:
        with open(path, "w") as f:
            for token_id in range(len(self.vocab)):
                if token_id in self.inverse_special:
                    continue
                var display = self.vocab[token_id]
                if display.byte_length() == 0:
                    continue
                var raw = ByteSequence(capacity=4)
                for cp in display.codepoints():
                    raw.append(Byte(self.cp_to_byte[Int(cp)]))
                var encoded = b64encode(Span[Byte](raw))
                f.write(encoded + " " + String(token_id) + "\n")

    def load_tiktoken(mut self, path: String) raises:
        var file_content: String
        with open(path, "r") as f:
            file_content = f.read()
        var raw_lines = file_content.split("\n")

        var mergeable_ranks = Dict[String, Int]()
        var all_tokens = List[ByteSequence]()
        var max_id = 0

        for line_ptr in raw_lines:
            var line = String(line_ptr.strip())
            if line.byte_length() == 0:
                continue
            var parts = line.split(" ")
            var raw = b64decode(parts[0])
            var rank = Int(parts[1])
            var key = BPETokenizer._bytes_key(Span[Byte](raw))
            mergeable_ranks[key] = rank
            while len(all_tokens) <= rank:
                all_tokens.append(ByteSequence())
            all_tokens[rank] = raw^
            if rank > max_id:
                max_id = rank

        var new_vocab_size = max_id + 1
        var new_vocab = Vocabulary(capacity=new_vocab_size)
        var new_table = TokenByteTable()
        new_table.reserve(new_vocab_size)
        for token_id in range(new_vocab_size):
            var raw_bytes = Span[Byte](all_tokens[token_id])
            var display = String(capacity=len(raw_bytes) * 3)
            for i in range(len(raw_bytes)):
                display += chr(self.byte_to_cp[Int(raw_bytes[i])])
            new_vocab.append(display)
            new_table.add(self._display_to_bytes(display))
        new_table.finish()

        self._recover_merges(mergeable_ranks, all_tokens)

        var new_lookup_table = MergeLookup()
        for merge in self.merges:
            new_lookup_table.set(merge.first, merge.second, merge.merged)

        # Populate byte_to_rank from the loaded file's rank assignments.
        # For SEQUENTIAL this usually differs from identity; for SHUFFLED
        # it matches the comptime LUT so the change is a no-op.
        var single_byte = ByteSequence()
        single_byte.resize(1, 0)
        for b in range(256):
            single_byte[0] = Byte(b)
            var key = BPETokenizer._bytes_key(
                Span[Byte](ptr=single_byte.unsafe_ptr(), length=1)
            )
            if key in mergeable_ranks:
                self.byte_to_rank[b] = mergeable_ranks[key]

        self.vocab = new_vocab^
        self.token_table = new_table^
        self.lookup_table = new_lookup_table^

        for item in Self.PT.special_tokens().items():
            if not item.value in self.inverse_special:
                self._register_special_token(item.key, item.value)


# ═══════════════════════════════════════════════════════════════════════════
# Hot-path helpers — raw pointer operations, zero allocations
# ═══════════════════════════════════════════════════════════════════════════


@always_inline
def merge_inplace(
    buf: UnsafePointer[Int, MutAnyOrigin],
    n: Int,
    a: Int,
    b: Int,
    m: Int,
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


# ── Convenience API: pre-built encodings ──────────────────────────

def _find_data_dir() raises -> String:
    """Locate the .tiktoken data files.

    Order: MBPE_DATA_DIR env var  →  ./data/  →  ../data/
    """
    var env_dir = getenv("MBPE_DATA_DIR", "")
    if env_dir.byte_length() > 0:
        return env_dir
    return "data"


struct Tokenizers:
    """Type-level enumeration of built-in encodings.

    Usage:
        var gpt2   = Tokenizers.get[Tokenizers.gpt2]()
        var cl100k = Tokenizers.get[Tokenizers.cl100k]()
        var o200k  = Tokenizers.get[Tokenizers.o200k]()
    """

    comptime gpt2 = GPT2Pretokenizer
    comptime cl100k = GPT4Pretokenizer[ByteMapping.SEQUENTIAL]
    comptime o200k = GPT4Pretokenizer[ByteMapping.SHUFFLED]

    @staticmethod
    def get[T: PreTokenizer](filename: String = "") raises -> BPETokenizer[T]:
        var fname = T.name() if filename.byte_length() == 0 else filename
        var tok = BPETokenizer[T]()
        tok.load_tiktoken(_find_data_dir() + "/" + fname + ".tiktoken")
        return tok^

    @staticmethod
    def train[T: PreTokenizer](
        corpus: Span[String, _], vocab_size: Int
    ) raises -> BPETokenizer[T]:
        var tok = BPETokenizer[T]()
        tok.train(corpus, vocab_size)
        return tok^
