"""Byte Pair Encoding tokenizer — train, encode, decode, save, load.

Design philosophy
-----------------
The hot path (training merges, encoding) works exclusively with Int token IDs.
Strings are materialized only when the outside world needs them: building the
vocabulary display strings, decoding IDs back to readable text, and serializing
to/from JSON.  This keeps allocations off the hot loop.

Byte-level base vocabulary (GPT-2 style)
-----------------------------------------
Instead of scanning the training corpus for unique characters, we start with
all 256 byte values (0x00–0xFF) as the base vocabulary.  Every Unicode
codepoint decomposes into 1–4 UTF-8 bytes, so every possible input is
representable.  There is no UNK token — ID 0 is simply byte 0x00.

Since raw bytes 0–255 can't live in a String (many aren't valid UTF-8), GPT-2
introduced a [`bytes_to_unicode`](https://github.com/openai/gpt-2/blob/master/src/encoder.py) table: printable bytes map to themselves and
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
    GPT2Pretokenizer,
    GPT4Pretokenizer,
    ByteMapping,
    WordCounts,
)
from bpe.array import IntArray, ByteArray, TokenSpan, ByteSpanArena


# ---------------------------------------------------------------------------
# MergeRule — a BPE merge: (a_id, b_id, merged_id)
# ---------------------------------------------------------------------------


@fieldwise_init
struct MergeRule(
    ImplicitlyCopyable
    & TrivialRegisterPassable
    & Hashable
    & Equatable
    & Writable
):
    var first: Int
    var second: Int
    var merged: Int

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

    _fast is a flat IntArray (1024 × 1024), index = (a << 10) | b;
    copies are deep (List has no refcounted sharing).  _slow is always
    owned (deep-copied).
    """

    var _fast: IntArray
    var _slow: Dict[Int, Int]

    def __init__(out self):
        self._fast = IntArray(length=CACHE_ENTRIES, fill=-1)
        self._slow = Dict[Int, Int]()

    def __init__(out self, *, copy: Self):
        self._fast = copy._fast.copy()
        self._slow = copy._slow.copy()

    def __init__(out self, *, deinit move: Self):
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
            String("MergeLookup(capacity=")
            + String(CACHE_ENTRIES)
            + String(")")
        )


# ---------------------------------------------------------------------------
# TokenByteTable — flat per-token byte storage
#
# Layout (delegated to a ByteSpanArena):
#   arena.bytes   : concatenated raw bytes of every token (flat allocation)
#   arena.spans   : per-token (offset, length) into `bytes`
#   token i's bytes live at bytes[spans[i].offset : spans[i].offset + spans[i].length]
#
# Memory model:
#   - The byte pool is a flat ByteArray (heap-backed, amortised growth),
#     exposing the unsafe_ptr() API the decode hot path uses.
#   - The span list is a List[TokenSpan] exposing the same unsafe_ptr()
#     raw-pointer access; each entry carries both bounds in one small
#     register-passable struct.
#   - Copies are deep (ImplicitlyCopyable).
#
# Invariants:
#   len(arena) == len(vocab) in BPETokenizer
#   no sentinel offset — callers read spans[id].length directly
# ---------------------------------------------------------------------------


struct TokenByteTable(ImplicitlyCopyable & Movable & Sized & Writable):
    var arena: ByteSpanArena

    def __init__(out self):
        self.arena = ByteSpanArena()

    def __init__(out self, *, copy: Self):
        """Deep copy of the byte pool and the span list."""
        self.arena = ByteSpanArena(copy=copy.arena)

    def __init__(out self, *, deinit move: Self):
        self.arena = move.arena^

    @always_inline
    def __len__(self) -> Int:
        return len(self.arena)

    def write_to[T: Writer](self, mut writer: T):
        """Write a short summary (token count, pool bytes) to `writer`."""
        writer.write(
            String("TokenByteTable(tokens=")
            + String(len(self.arena))
            + String(", bytes=")
            + String(len(self.arena.bytes))
            + String(")")
        )

    @always_inline
    def reserve(mut self, max_tokens: Int):
        self.arena.spans.reserve(max_tokens)

    @always_inline
    def add[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, raw: Span[Byte, origin]):
        """Append a token's raw bytes."""
        _ = self.arena.add(raw)

    @always_inline
    def set_bytes[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, id: Int, raw: Span[Byte, origin]):
        """Register a token at an exact id, padding gaps with empty tokens."""
        while len(self.arena) <= id:
            self.arena.spans.append(TokenSpan(len(self.arena.bytes), 0))
        var off = len(self.arena.bytes)
        self.arena.bytes.reserve(off + len(raw))
        self.arena.bytes.extend(raw)
        self.arena.spans[id] = TokenSpan(off, len(raw))


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
# Conclusion: `merges` is metadata/serialization only — not on the encode
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

# ---------------------------------------------------------------------------

struct BPETokenizer[PT: PreTokenizer = GPT2Pretokenizer](
    Sized & Movable & Writable
):
    """A Byte-Pair Encoding tokenizer, parameterized over a pluggable
    pre-tokenizer (PT) that decides how raw text is first split into
    words before BPE merging runs on those words.

    High-level lifecycle:
        1. Construct: __init__ sets up the fixed byte<->safe-Unicode
           mapping (GPT-2 style) that every other method relies on.
        2. (Optional) register_special_tokens: reserve IDs for tokens
           that should never be split by BPE (e.g. "<|endoftext|>").
        3. train(): learn merges from a corpus, building up `merges` and
           the raw per-token byte table from the base 256 byte tokens to
           `vocab_size`.
        4. encode_ordinary() / decode (elsewhere): use the learned
           `lookup_table` to tokenize new text, or `token_table` to
           reconstruct raw bytes from token IDs.

    A token's identity lives in exactly ONE place: `token_table` maps
    token id -> raw bytes.  The safe-Unicode *display* form (what a
    tiktoken-format file shows) is derived on demand from those bytes via
    `byte_to_cp` (see display_of()); it is never stored.
    """

    var pt: Self.PT
    """The pre-tokenizer instance (e.g. GPT2Pretokenizer,
    GPT4Pretokenizer) -- decides how raw text is split into words before
    BPE operates on them, and how bytes map to base token IDs (see
    Self.PT.byte_to_id / id_to_byte, used throughout training/encoding)."""

    var merges: List[MergeRule]
    """The ordered list of merges learned during training, in the order
    they were learned (earliest merge = lowest rank). Each rule records
    which two token IDs were merged and what new ID they became."""

    var lookup_table: MergeLookup
    """O(1) lookup: given a pair of adjacent token IDs, what merged ID
    (if any) do they collapse to, and at what rank (lower rank = learned
    earlier = higher priority to apply first). Built during train(),
    used during encode_ordinary() to avoid rescanning `merges` linearly."""

    var byte_to_cp: Dict[Int, Int]
    """Raw byte value (0-255) -> Unicode codepoint used to *display* that
    byte. Most printable ASCII/Latin-1 bytes map to themselves; bytes that
    aren't safely printable/whitespace-safe get remapped to a codepoint
    at 256+ instead (see __init__ for the exact GPT-2 bytes_to_unicode
    rule). This is what makes display strings always safe to print/write
    to a file, even though they represent arbitrary binary data.

    The mapping is a bijection (every raw byte maps to a unique
    codepoint and vice versa), which is what lets display_of() round-trip
    raw bytes without a reverse table."""

    var byte_to_rank: IntArray
    """byte value (0-255) -> base token ID for that byte, indexed
    directly (byte_to_rank[byte] = id). For most pre-tokenizers this is
    just the identity (byte_to_rank[b] == b, i.e. SEQUENTIAL byte
    mapping), but some PT implementations shuffle the byte->ID
    assignment (ByteMapping.SHUFFLED), in which case this array (or
    Self.PT.byte_to_id directly) is the source of truth instead of
    assuming byte == id."""

    var token_table: TokenByteTable
    """token id -> raw bytes (the single source of truth for what a token
    decodes to). Used to decode token IDs back to real bytes, to write
    .tiktoken files, and to derive display strings. Built incrementally
    during train() as each new token (base byte or merge) is registered."""

    var special_bytes: Dict[String, Int]
    """Special-token text -> reserved token ID. Special tokens (e.g.
    "<|endoftext|>") bypass BPE splitting entirely during encoding --
    they're matched and emitted as a single ID."""

    var inverse_special: Dict[Int, String]
    """The inverse of special_bytes: reserved token ID -> special-token
    text. Used wherever a token ID needs to be resolved back to its
    special-token string (e.g. during decode)."""

    def __init__(out self):
        """Construct an empty, untrained tokenizer and initialize the
        fixed GPT-2-style byte<->safe-Unicode display mapping.

        This mapping never changes after construction (it's independent
        of any training corpus) -- it's purely about choosing a safe,
        printable Unicode codepoint to *represent* each of the 256
        possible byte values, so that vocab strings are always valid to
        print, log, or write to a tiktoken-format file, even for bytes
        that are normally unprintable (control characters) or otherwise
        awkward (leading/trailing whitespace, which some tools trim or
        mangle if left as literal spaces).
        """
        self.pt = Self.PT()
        self.merges = List[MergeRule]()
        self.lookup_table = MergeLookup()
        self.byte_to_cp = Dict[Int, Int]()
        # Default identity mapping (byte b -> base token id b); overwritten
        # per-rank in train() step 3 if Self.PT uses a SHUFFLED byte
        # mapping instead of SEQUENTIAL.
        self.byte_to_rank = [b for b in range(256)]
        self.token_table = TokenByteTable()
        self.special_bytes = Dict[String, Int]()
        self.inverse_special = Dict[Int, String]()
        # GPT-2 bytes_to_unicode mapping (fixed at init, used by all methods)
        #
        # The idea: bytes 0x21-0x7E (printable ASCII), 0xA1-0xAC, and
        # 0xAE-0xFF (printable Latin-1 supplement, skipping the soft-
        # hyphen 0xAD) are already safe to display as themselves -- map
        # each straight to the identical codepoint.
        #
        # Every OTHER byte (control characters, space 0x20, 0x7F-0xA0,
        # 0xAD) is NOT safe to leave as a literal character in a display
        # string (could be invisible, get trimmed, or break formats that
        # treat it specially). Each of these gets assigned an arbitrary
        # *unused* codepoint starting at 256 instead -- codepoints 256+
        # are guaranteed not to collide with any of the printable-byte
        # codepoints assigned above (which are all <= 0xFF), so the
        # mapping stays one-to-one and reversible via cp_to_byte.
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
            else:
                # Non-printable byte: assign the next free codepoint at
                # 256+n, incrementing n for each one so no two
                # non-printable bytes ever collide.
                var cp = 256 + n
                self.byte_to_cp[b] = cp
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
        """Register one special token, recording it in every relevant
        table: special_bytes/inverse_special (for encode/decode lookups),
        and token_table (so it decodes to its own literal bytes -- special
        tokens skip the bytes_to_unicode detour entirely, since their raw
        bytes ARE their display text). set_bytes() grows token_table with
        empty placeholder slots if `id` lies beyond its current size.

        Raises:
            Error: if text is empty, or this exact text was already
                registered as a special token.
        """
        if text.byte_length() == 0:
            raise Error("special token text must not be empty")
        if text in self.special_bytes:
            raise Error("duplicate special token: " + text)
        self.special_bytes[text] = id
        self.inverse_special[id] = text
        self.token_table.set_bytes(id, text.as_bytes())

    # ── training ────────────────────────────────────────────────────────
    # The algorithm:
    #   1. Pre-tokenise the corpus and count word frequencies.
    #   2. Build the byte→safe-unicode mapping (GPT-2 style).
    #   3. Initialise the base tokens: the 256 byte values at IDs 0–255.
    #   4. Split every word into a list of base token IDs (one per byte);
    #      each distinct word is stored ONCE with its frequency.
    #   5. Repeatedly find the most frequent adjacent pair and merge it,
    #      appending the new token's raw bytes to the token table.
    #
    # Step 5 is the core BPE loop.  Each merge:
    #   - records the pair (a_id, b_id) and the new merged_id in `merges`
    #   - replaces every occurrence of a_id followed by b_id with merged_id
    #   - appends the concatenated raw bytes of its two parents to
    #     `token_table` (pure byte concatenation -- no display round-trip)
    #
    # The loop stops when we reach vocab_size or when no pairs remain
    # (every word has been reduced to a single token).
    #
    # Design B (see SYSTEM.md 4.1.9a): words live in a flat IntArray arena
    # (no SEP, no replication, one allocation per training run); the `where_dict`
    # map tracks pair → affected word indices, so each merge scans only the
    # words that contain the pair, compacting them in place, and pair counts
    # are updated arithmetically with delta × word frequency instead of
    # rescanning the whole corpus.
    #
    # Pair encoding: throughout training, a pair of adjacent token IDs
    # (a_id, b_id) is packed into a single Int dict key via
    # `(a_id << ENCODE_SHIFT) | b_id`, so `stats` (pair -> total weighted
    # frequency) and `where_dict` (pair -> word indices containing it) can
    # both be plain Dict[Int, ...] instead of needing a 2-tuple key type.
    # This only works correctly as long as every token ID fits within
    # ENCODE_SHIFT bits -- i.e. vocab_size must stay under 2^ENCODE_SHIFT,
    # or two different (a_id, b_id) pairs could collide onto the same
    # packed key.
    # ─────────────────────────────────────────────────────────────────────

    def train[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, corpus: Span[String, origin], vocab_size: Int) raises:
        """Learn a BPE vocabulary of size `vocab_size` from `corpus`.

        Populates self.merges, self.lookup_table, and self.token_table
        from scratch (any prior training state is discarded). Base byte
        tokens always occupy IDs 0-255; merge tokens are appended
        afterward in the order they're learned.

        Args:
            corpus: The training texts.
            vocab_size: Target vocabulary size. Training stops early if
                no mergeable pairs remain before reaching this size.

        Raises:
            Error: if vocab_size < 256 (too small to hold the base byte
                vocabulary alone).
        """
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

        # ---- 3. Initialise base token storage ----------------------------
        # IDs 0–255 are the 256 byte values, each stored as its single raw
        # byte.  Merge tokens are appended below.
        # For SEQUENTIAL: rank == byte, so token rank holds byte rank.
        # For SHUFFLED:   rank != byte, so we use id_to_byte(rank) to find
        # the raw byte for each rank, ensuring token rank holds the right
        # byte value.
        self.token_table = TokenByteTable()
        self.token_table.reserve(vocab_size)
        self.byte_to_rank = [b for b in range(256)]
        for rank in range(256):
            # id_to_byte(rank) is the identity for SEQUENTIAL pre-
            # tokenizers (rank IS the byte), but may differ for
            # SHUFFLED ones -- this indirection is what makes both
            # byte-mapping schemes work through the same loop.
            var b = Self.PT.id_to_byte(rank)
            var raw = ByteArray(capacity=1)
            raw.append(Byte(b))
            self.token_table.add(Span[Byte](raw))

        # ---- 4. Build per-word token-ID storage + initial pair counts -------
        # Design B (4.1.9a): each distinct word is stored ONCE in a flat
        # token arena (word_freqs iteration order) with its frequency in
        # word_freq; there is no SEP and no replication — frequency
        # weighting is applied arithmetically as delta × freq.  `where_dict`
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

        # First pass: sum every word's byte-length (weighted by nothing --
        # this is total DISTINCT-word length, used only to size `arena`
        # for one single allocation up front, since arena stores each
        # distinct word's tokens once regardless of its frequency).
        var total_tokens: Int = 0
        for ei in range(word_counts.n_entries):  # ei = entry index into word_counts
            total_tokens += word_counts.arena.spans[ei].length
        var arena = IntArray(capacity=total_tokens)
        var word_offs = IntArray(
            capacity=word_counts.n_entries
        )  # word index -> start offset in arena
        var word_len = IntArray(
            capacity=word_counts.n_entries
        )  # word index -> current token count (shrinks as merges compact it)
        var word_freq = IntArray(
            capacity=word_counts.n_entries
        )  # word index -> corpus frequency (how many times this word occurred)
        var stats = Dict[
            Int, Int
        ]()  # packed pair key -> total frequency-weighted occurrence count
        var where_dict = Dict[
            Int, IntArray
        ]()  # packed pair key -> list of word indices that currently contain this pair
        var wb = word_counts.arena.bytes.unsafe_ptr()
        for ei in range(word_counts.n_entries):  # ei = entry index (same as above)
            var off = word_counts.arena.spans[ei].offset
            var ln = word_counts.arena.spans[ei].length
            var freq = word_counts.counts[ei]
            var off_arena = len(arena)
            word_offs.append(off_arena)
            # Convert this word's raw bytes into base token IDs and lay
            # them into the shared arena, back-to-back with every other
            # word (no separator needed -- word_offs/word_len bound each
            # word's slice explicitly).
            for i in range(ln):
                arena.append(Self.PT.byte_to_id(Int(wb[off + i])))
            word_len.append(ln)
            word_freq.append(freq)
            var iw = len(word_offs) - 1  # iw = index of this word
            # Seed initial pair statistics: every adjacent pair within
            # this word contributes `freq` to that pair's total count
            # (since this word-shape occurs `freq` times in the corpus),
            # and this word index is recorded as one of the pair's
            # "affected words" for the merge loop to consult later.
            for i in range(off_arena, off_arena + ln - 1):
                var key = (arena[i] << ENCODE_SHIFT) | arena[i + 1]
                stats[key] = stats.get(key, 0) + freq
                if key not in where_dict:
                    where_dict[key] = IntArray()
                where_dict[key].append(iw)

        # ---- 5. Merge loop (per-word where_to_update) ----------------------
        # Each iteration finds the most frequent pair, then scans ONLY the
        # words known to contain it, compacting them in place with the same
        # i += 2 / last-emitted-token semantics as the flat design, and
        # adjusts `stats` incrementally (the 5 pairs affected per
        # occurrence, × word frequency).
        self.lookup_table = MergeLookup()
        self.merges = List[MergeRule]()
        while len(self.token_table) < vocab_size:
            # Find the most frequent pair.
            # (Linear scan over `stats` each iteration -- straightforward
            # but O(distinct pairs) per merge; the payoff of where_dict is
            # in step 5's per-word work below, not in this selection step.)
            var best_pair: Tuple[Int, Int] = (0, 0)
            var max_freq = -1
            for item in stats.items():
                if item.value > max_freq:
                    max_freq = item.value
                    best_pair = (
                        item.key >> ENCODE_SHIFT,
                        item.key & ENCODE_MASK,
                    )
            if max_freq <= 0:
                # No mergeable pairs remain -- every word has been reduced
                # to a single token.  Stop early even if vocab_size
                # wasn't reached.
                break

            var a_id = best_pair[0]
            var b_id = best_pair[1]
            var best_key = (a_id << ENCODE_SHIFT) | b_id
            var merged_id = len(self.token_table)

            # Snapshot the affected-word list; the same key's list is never
            # appended to during its own merge (created pairs always involve
            # the brand-new merged_id), so the snapshot length is stable.
            var snap = len(where_dict[best_key])
            for wi in range(snap):  # wi = word-loop index (into snapshot)
                var iw = where_dict[best_key][wi]  # iw = index of word
                var freq = word_freq[iw]
                var start = word_offs[iw]
                var n = word_len[iw]
                var wt = arena.unsafe_ptr() + start
                # In-place left-to-right compaction: `w` is the write
                # cursor (where the next surviving/merged token gets
                # written), `i` is the read cursor. Because writes never
                # get ahead of reads (w <= i always), this can safely
                # compact the word's slice of `arena` in place with no
                # extra buffer.
                var w = 0
                var i = 0
                while i < n:
                    if i < n - 1 and wt[i] == a_id and wt[i + 1] == b_id:
                        # Found an occurrence of the target pair at
                        # position i. Before collapsing it, update
                        # `stats` for every pair that's about to be
                        # destroyed or created by this merge:
                        #
                        # Decrement destroyed pairs: (prev, a), (a, b),
                        # (b, next); prev is the last emitted token.
                        if w > 0:
                            # (prev_token, a_id) is destroyed -- prev_token
                            # will now be adjacent to merged_id instead.
                            var pk = (wt[w - 1] << ENCODE_SHIFT) | wt[i]
                            if pk in stats:
                                var nv = stats[pk] - freq  # nv = new count
                                stats[pk] = nv if nv > 0 else 0
                        # (a_id, b_id) itself is destroyed -- this is the
                        # pair being merged away.
                        var mk = (wt[i] << ENCODE_SHIFT) | wt[i + 1]
                        if mk in stats:
                            var nv = stats[mk] - freq
                            stats[mk] = nv if nv > 0 else 0
                        if i + 2 < n:
                            # (b_id, next_token) is destroyed -- next_token
                            # will now be adjacent to merged_id instead.
                            var nk = (wt[i + 1] << ENCODE_SHIFT) | wt[i + 2]
                            if nk in stats:
                                var nv = stats[nk] - freq
                                stats[nk] = nv if nv > 0 else 0
                        # Increment created pairs: (prev, new_id),
                        # (new_id, next), and register the word.
                        if w > 0:
                            # New pair (prev_token, merged_id) now exists
                            # -- record it and note this word contains it,
                            # so a FUTURE merge on this new pair can find
                            # this word via where_dict without a corpus
                            # rescan.
                            var pk2 = (wt[w - 1] << ENCODE_SHIFT) | merged_id
                            stats[pk2] = stats.get(pk2, 0) + freq
                            if pk2 not in where_dict:
                                where_dict[pk2] = IntArray()
                            where_dict[pk2].append(iw)
                        if i + 2 < n:
                            # New pair (merged_id, next_token) now exists
                            # -- same bookkeeping as above.
                            var nk2 = (merged_id << ENCODE_SHIFT) | wt[i + 2]
                            stats[nk2] = stats.get(nk2, 0) + freq
                            if nk2 not in where_dict:
                                where_dict[nk2] = IntArray()
                            where_dict[nk2].append(iw)
                        # Collapse the pair: write the single merged_id in
                        # place of the two source tokens, advance the read
                        # cursor past BOTH consumed tokens.
                        wt[w] = merged_id
                        w += 1
                        i += 2
                    else:
                        # No match at this position -- just copy the
                        # token forward (compaction may still shift its
                        # position left if earlier merges shrank the word).
                        wt[w] = wt[i]
                        w += 1
                        i += 1
                # Word is now shorter (or unchanged if the pair didn't
                # actually occur here despite being in where_dict's list
                # -- can't happen given how where_dict is maintained, but
                # word_len is updated regardless as the new authoritative
                # length).
                word_len[iw] = w

            # Record the merge.
            self.merges.append(MergeRule(a_id, b_id, merged_id))
            self.lookup_table.set(a_id, b_id, merged_id)

            # Build the flat byte storage for the new token.  A merge
            # token's bytes are simply its two parents' raw bytes
            # concatenated (pure byte concat -- no display round-trip,
            # no spacer collapse, since nothing is re-encoded).
            var spans = self.token_table.arena.spans.unsafe_ptr()
            var pool = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
            var la = spans[a_id].length
            var lb = spans[b_id].length
            var merged_bytes = ByteArray(capacity=la + lb)
            merged_bytes.resize(la + lb, 0)
            memcpy(dest=merged_bytes.unsafe_ptr(), src=pool + spans[a_id].offset, count=la)
            memcpy(dest=merged_bytes.unsafe_ptr() + la, src=pool + spans[b_id].offset, count=lb)
            self.token_table.add(Span[Byte](merged_bytes))

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
    #
    # Two different merge strategies are used depending on word length
    # (see SCAN_LIMIT below):
    #   - Short words: a simple O(n^2) scan-based greedy merge, which is
    #     cheaper in practice than heap bookkeeping overhead at small n.
    #   - Long words: an O(N log N) heap-driven merge over a doubly
    #     linked list of token nodes, which pays off once n is large
    #     enough that repeated O(n) scans would dominate.
    # ─────────────────────────────────────────────────────────────────────

    def encode_ordinary[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> IntArray:
        """Encode `text` into token IDs using the learned BPE merges.

        Does NOT handle special tokens (see the full encode() elsewhere
        for that) -- this only applies ordinary byte-level BPE merging.

        Args:
            text: The text to encode. Any string-like view works, no
                ownership required.

        Returns:
            The sequence of token IDs representing `text`.
        """
        if text.byte_length() == 0:
            return IntArray()
        # ---- 1. Pre-tokenise into words (zero-copy views) --------------
        ref words = self.pt.split_view(text)
        # ---- 2. Compute total bytes for a single allocation -------------
        # Upper bound on the output token count is the total input byte
        # count (encoding never produces MORE tokens than input bytes,
        # only fewer via merging) -- so this sizes `result` once instead
        # of growing it dynamically per word.
        var total_bytes = 0
        for word in words:
            total_bytes += word.byte_length()
        if total_bytes == 0:
            return IntArray()

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
        var result = IntArray()
        result.resize(total_bytes, 0)
        var write_pos = 0
        # Scratch buffers for the heap-driven (long-word) path only,
        # reused across every long word in this call -- reallocated only
        # when a longer word than previously seen shows up (see `cap`
        # below), so most words hit these buffers "for free".
        var ids_buf = alloc[Int](
            0
        )  # node index -> current token id (updates in place as nodes merge)
        var nxt_buf = alloc[Int](
            0
        )  # node index -> next node index in the linked list (-1 = end)
        var prv_buf = alloc[Int](
            0
        )  # node index -> previous node index in the linked list (-1 = start)
        var alive_buf = alloc[UInt8](
            0
        )  # node index -> 1 if this node is still part of the list, 0 if merged away
        var cap = 0  # current allocated capacity of the four buffers above
        var heap = BinaryHeap[Int]()
        var btr = self.byte_to_rank.unsafe_ptr()
        for word in words:
            var ptr = word.unsafe_ptr()
            var n = word.byte_length()
            var dst = result.unsafe_ptr() + write_pos

            # Copy bytes as Ints (via PT byte mapping)
            # Trivial case: 0 or 1 byte can't have any adjacent pair to
            # merge, so just copy the base token ID(s) straight through.
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
                # Repeatedly find and apply the single lowest-rank
                # mergeable pair anywhere in the word, one merge per
                # outer-loop iteration, until no pair in lookup_table
                # matches anymore. This full O(len) rescan per merge is
                # the "O(n^2) is cheaper" tradeoff mentioned above --
                # acceptable because `len` is small here by construction
                # (n < SCAN_LIMIT).
                while len >= 2:
                    var best_rank = -1
                    var best_a = -1
                    var best_b = -1
                    var best_m = -1
                    for i in range(len - 1):
                        var merged = self.lookup_table.get(dst[i], dst[i + 1])
                        if merged >= 0 and (
                            best_rank < 0 or merged < best_rank
                        ):
                            # Lower merged-id here means "learned earlier
                            # during training" (ids are assigned in merge
                            # order starting at 256), i.e. higher merge
                            # priority -- so tracking the minimum directly
                            # gives the correct greedy choice.
                            best_rank = merged
                            best_a = dst[i]
                            best_b = dst[i + 1]
                            best_m = merged
                    if best_rank < 0:
                        # No pair anywhere in the word matches any known
                        # merge rule -- fully reduced, stop.
                        break
                    len = merge_inplace(dst, len, best_a, best_b, best_m)
                write_pos += len
                continue

            # Long words: heap-driven merge (O(N log N)).
            if n > cap:
                # Current scratch buffers are too small for this word --
                # grow them (freeing the old, smaller ones first). Sized
                # exactly to `n` rather than some padded amount, so this
                # only reallocates when a strictly longer word than any
                # seen so far arrives.
                ids_buf.free()
                nxt_buf.free()
                prv_buf.free()
                alive_buf.free()
                ids_buf = alloc[Int](n)
                nxt_buf = alloc[Int](n)
                prv_buf = alloc[Int](n)
                alive_buf = alloc[UInt8](n)
                cap = n

            # Initialize one linked-list node per input byte: token id,
            # next/prev pointers forming a simple doubly linked chain
            # 0 <-> 1 <-> 2 <-> ... <-> n-1, and mark every node alive.
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
            # Each heap entry packs (rank, left-node-index) into one Int
            # via `rank << HEAP_SHIFT | i`, then negates it -- since
            # BinaryHeap is a max-heap, pushing the negated key means the
            # eventual max-heap pop returns the numerically LOWEST rank
            # first (lowest rank = learned earliest = correct greedy
            # priority), with ties broken by lowest node index (since a
            # smaller `i` makes the packed value more negative / "more
            # max" after negation, matching the left-to-right tie-break
            # used by the scan-based short-word path above).
            for i in range(n - 1):
                var r0 = self.lookup_table.get(ids_buf[i], ids_buf[i + 1])
                if r0 >= 0:
                    heap.push(-(r0 << HEAP_SHIFT | i))

            # Greedy lowest-rank merge (lazy-validated heap).
            #
            # "Lazy" because entries aren't removed from the heap when
            # they become stale (e.g. one of their two nodes got merged
            # into something else by an earlier pop) -- instead, every
            # popped entry is REVALIDATED against the current state
            # before being trusted, and simply skipped if it's no longer
            # accurate. This avoids the cost of an indexed/decrease-key
            # heap at the price of some extra (harmless) stale pops.
            while len(heap) > 0:
                var key = -heap.pop()  # undo the negation from push
                var e = key & HEAP_MASK  # recover the left-node index
                var rank = key >> HEAP_SHIFT  # recover the rank
                if alive_buf[e] == 0:
                    # Node e was already merged away by an earlier,
                    # higher-priority pop -- this entry is stale, skip.
                    continue
                var j = nxt_buf[e]
                if j < 0:
                    # e has no right neighbor anymore (e.g. it was at the
                    # end, or its old neighbor already merged elsewhere)
                    # -- nothing to merge with, skip.
                    continue
                # Revalidate: the pair may have changed since it was pushed.
                # (e.g. j itself might have merged with something to its
                # right already, changing what e's current right-neighbor
                # pair actually is)
                if self.lookup_table.get(ids_buf[e], ids_buf[j]) != rank:
                    continue

                # Merge: node e absorbs node j (e stays, j is spliced out).
                # e's token id becomes the merged id; j is marked dead and
                # unlinked from the list (e's next now skips past j to
                # whatever came after j).
                ids_buf[e] = rank
                var k = nxt_buf[j]
                if k >= 0:
                    prv_buf[k] = e
                nxt_buf[e] = k
                alive_buf[j] = 0

                # Only the two pairs touching the merge site can change
                # as a result of this merge (e's new pair with its
                # left neighbor p, and e's new pair with its new right
                # neighbor k) -- every other pair in the word is
                # unaffected, so only these two get freshly pushed.
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
            # Node 0 always survives as a list head (it can only ever be
            # the "e" side of a merge, absorbing neighbors, never the "j"
            # side that gets removed) -- so walking forward from 0 via
            # nxt_buf visits every surviving node in final left-to-right
            # order.
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

        # Trim to actual used size (write_pos is usually less than the
        # total_bytes upper bound allocated earlier, since merging
        # reduces token count below the raw byte count for any text with
        # learned merge patterns).
        result.resize(write_pos, 0)
        return result^

    def encode[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> IntArray:
        """Encode `text` into token IDs, handling special tokens.

        Unlike encode_ordinary(), this scans for registered special
        tokens (e.g. "<|endoftext|>") and emits them as single reserved
        IDs wherever they appear literally in the text, running ordinary
        BPE encoding on everything in between.

        Args:
            text: The text to encode.

        Returns:
            The sequence of token IDs representing `text`, with special
            tokens emitted as single IDs and everything else BPE-encoded.
        """
        # Fast path: no special tokens registered at all, so there's
        # nothing to scan for -- skip straight to ordinary encoding.
        if len(self.special_bytes) == 0:
            return self.encode_ordinary(text)

        var n = text.byte_length()
        if n == 0:
            return IntArray()

        var bytes = text.as_bytes()
        var result = IntArray()
        var pos = 0

        # Walk the text left to right. At each position, first check
        # whether a special token starts exactly HERE (byte-for-byte);
        # if not, find the nearest special token occurrence anywhere
        # ahead and BPE-encode everything up to it in one batch.
        while pos < n:
            # ---- Step A: does a special token start at exactly `pos`? ----
            var found_id = -1
            var found_len = 0
            for item in self.special_bytes.items():
                var tok = item.key
                var tok_id = item.value
                var tok_len = tok.byte_length()
                if pos + tok_len <= n:
                    # Manual byte-for-byte comparison (special tokens are
                    # typically short, e.g. "<|endoftext|>", so this is
                    # cheap even without a specialized string-match
                    # routine).
                    var matched = True
                    var tok_bytes = tok.as_bytes()
                    for k in range(tok_len):
                        if bytes[pos + k] != tok_bytes[k]:
                            matched = False
                            break
                    if matched:
                        found_id = tok_id
                        found_len = tok_len
                        # NOTE: stops at the FIRST matching special token
                        # in dict iteration order (insertion order), not
                        # necessarily the LONGEST match if multiple
                        # special tokens could match at this position
                        # (e.g. one token being a prefix of another). If
                        # that's a real possibility for your special-token
                        # set, registration order determines priority --
                        # worth being deliberate about registration order,
                        # or extending this to prefer the longest match.
                        break
            if found_id >= 0:
                # A special token matched right here -- emit its single
                # reserved ID and jump past it entirely (BPE never sees
                # its bytes).
                result.append(found_id)
                pos += found_len
            else:
                # ---- Step B: no special token starts exactly at `pos` --
                # find the NEAREST special token occurrence anywhere
                # ahead (if any), and BPE-encode the plain-text segment
                # up to that point in one batch (much cheaper than
                # re-running Step A one byte at a time until we hit it).
                var start = pos
                var next_special = n  # default: no special token found ahead -> segment runs to end of text
                for item in self.special_bytes.items():
                    var tok = item.key
                    var found_at = text.find(tok, start)
                    if found_at >= 0 and found_at < next_special:
                        next_special = found_at
                if next_special > start:
                    # Found a plain-text segment [start, next_special) to
                    # BPE-encode as one unit before hitting the next
                    # special token (or the end of text).
                    var seg = StringSlice(
                        unsafe_from_utf8=bytes[start:next_special]
                    )
                    for id in self.encode_ordinary(seg):
                        result.append(id)
                    pos = next_special
                elif next_special == start:
                    # Defensive fallback: text.find() reports a special
                    # token starting exactly at `start`, yet Step A's
                    # direct byte comparison just failed to match
                    # anything here. This shouldn't happen if find() and
                    # the manual byte comparison agree on what "matches"
                    # means, but as a safety net against that
                    # inconsistency, advance by a single byte rather than
                    # looping forever at the same position.
                    pos += 1
                else:
                    # next_special < start can't happen given `start` is
                    # the search floor passed to find() -- this branch is
                    # effectively "no special token found anywhere ahead"
                    # (next_special == n, i.e. not > start only if
                    # start == n, but the outer while already guards
                    # pos < n) -- encode the remaining tail as one final
                    # plain-text segment.
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
    #
    # The core technique shared by decode()/decode_bytes()/
    # decode_with_offsets(): two passes over `ids`. Pass 1 validates every
    # ID and sums up the total output byte length, so the destination
    # buffer can be allocated ONCE at the exact right size. Pass 2 does
    # the actual memcpy of each token's raw bytes into that buffer at the
    # right offset. This avoids both a growable-buffer's repeated
    # reallocation AND any risk of writing out of bounds from a
    # miscalculated size.
    # ─────────────────────────────────────────────────────────────────────

    def decode[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, ids: Span[Int, origin]) raises -> String:
        """Decode a sequence of token IDs back into text.

        Args:
            ids: The token IDs to decode.

        Returns:
            The reconstructed text.

        Raises:
            Error: if any ID in `ids` is negative or >= the vocabulary
                size.
        """
        if len(ids) == 0:
            return String("")
        var spans = self.token_table.arena.spans.unsafe_ptr()
        var n_tokens = len(self.token_table)
        # ---- Pass 1: validate every ID and compute total output size ----
        var total: Int = 0
        for id in ids:
            if id < 0 or id >= n_tokens:
                raise Error("token ID out of range: " + String(id))
            total += spans[id].length
        if total == 0:
            return String("")
        # Allocate the exact-size destination String UNINITIALIZED (its
        # bytes are garbage until written below) -- this avoids
        # zero-filling memory that's about to be fully overwritten anyway.
        var result = String(unsafe_uninit_length=total)
        # Get a raw, mutable pointer into the String's own byte buffer.
        # This is writing directly into String-owned memory rather than
        # building a separate buffer and copying it in afterward --
        # `unsafe_mut_cast[True]()` is what makes this legal despite
        # as_bytes() normally returning an immutable view; safe here only
        # because `result` was just uninitialized-allocated above and
        # nothing else holds a reference to it yet.
        var dst = result.as_bytes().unsafe_ptr().unsafe_mut_cast[True]()
        var ptr = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
        # ---- Pass 2: copy each token's raw bytes into place -----------
        var write_offset: Int = 0
        for id in ids:
            var n = spans[id].length
            if n > 0:
                memcpy(
                    dest=dst + write_offset,
                    src=ptr + spans[id].offset,
                    count=n,
                )
                write_offset += n
        return result^

    def __len__(self) -> Int:
        """Return the current vocabulary size (number of trained/loaded
        tokens, base bytes + merges + special tokens)."""
        return len(self.token_table)

    def name(self) -> String:
        """Return the pre-tokenizer's name (e.g. identifying which
        PreTokenizer variant -- GPT2Pretokenizer, GPT4Pretokenizer --
        this tokenizer was built with)."""
        return Self.PT.name()

    def decode_bytes[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, ids: Span[Int, origin]) raises -> ByteArray:
        """Decode a sequence of token IDs back into raw bytes (as opposed
        to decode(), which returns a String). Useful when the decoded
        output isn't guaranteed to be valid UTF-8 text on its own, or
        when the caller wants to avoid String's UTF-8 assumptions
        entirely. Same two-pass validate-then-copy approach as decode().

        Args:
            ids: The token IDs to decode.

        Returns:
            The reconstructed raw bytes.

        Raises:
            Error: if any ID in `ids` is negative or >= the vocabulary
                size.
        """
        if len(ids) == 0:
            return ByteArray()
        var spans = self.token_table.arena.spans.unsafe_ptr()
        var n_tokens = len(self.token_table)
        var total: Int = 0
        for id in ids:
            if id < 0 or id >= n_tokens:
                raise Error("token ID out of range: " + String(id))
            total += spans[id].length
        if total == 0:
            return ByteArray()
        var result = ByteArray(capacity=total)
        result.resize(total, 0)
        var dst = result.unsafe_ptr()
        var src = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
        var write_offset: Int = 0
        for id in ids:
            var n = spans[id].length
            if n > 0:
                memcpy(
                    dest=dst + write_offset,
                    src=src + spans[id].offset,
                    count=n,
                )
                write_offset += n
        return result^

    def decode_single_token_bytes(self, id: Int) raises -> ByteArray:
        """Return the raw bytes for exactly one token ID, without going
        through the multi-token decode machinery. Useful for inspecting
        or debugging what a single token actually represents.

        Args:
            id: The token ID to look up.

        Returns:
            That token's raw bytes (empty if the token has zero length).

        Raises:
            Error: if id is negative or >= the vocabulary size.
        """
        if id < 0 or id >= len(self.token_table):
            raise Error("token ID out of range: " + String(id))
        var spans = self.token_table.arena.spans.unsafe_ptr()
        var n = spans[id].length
        if n == 0:
            return ByteArray()
        var off = spans[id].offset
        var ptr = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
        var result = ByteArray(capacity=n)
        result.resize(n, 0)
        memcpy(dest=result.unsafe_ptr(), src=ptr + off, count=n)
        return result^

    def decode_with_offsets[
        mut: Bool, //, origin: Origin[mut=mut]
    ](
        self, ids: Span[Int, origin], mut starts: IntArray, mut ends: List[Int]
    ) raises -> String:
        """Decode `ids` into text exactly like decode(), but ALSO records
        each token's byte-offset span within the resulting string, via
        the caller-supplied `starts`/`ends` output arrays.

        After this call, for the i-th token in `ids`: the substring of
        the returned text from starts[i] to ends[i] is exactly that
        token's decoded bytes. Useful for mapping tokens back to
        positions in the reconstructed text (e.g. for highlighting which
        substring a given token corresponds to).

        Args:
            ids: The token IDs to decode.
            starts: Output -- appended with each token's starting byte
                offset in the returned string, in the same order as
                `ids`. Cleared state is the caller's responsibility;
                this method only appends.
            ends: Output -- appended with each token's ending byte
                offset (exclusive) in the returned string, in the same
                order as `ids`.

        Returns:
            The reconstructed text (identical to what decode() would
            return for the same `ids`).

        Raises:
            Error: if any ID in `ids` is negative or >= the vocabulary
                size.
        """
        if len(ids) == 0:
            return String("")
        var spans = self.token_table.arena.spans.unsafe_ptr()
        var n_tokens = len(self.token_table)
        var total: Int = 0
        for id in ids:
            if id < 0 or id >= n_tokens:
                raise Error("token ID out of range: " + String(id))
            total += spans[id].length
        if total == 0:
            return String("")
        var result = String(unsafe_uninit_length=total)
        var dst = result.as_bytes().unsafe_ptr().unsafe_mut_cast[True]()
        var ptr = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
        var write_offset: Int = 0
        for id in ids:
            var n = spans[id].length
            # Record the span BEFORE writing this token's bytes (start)
            # and AFTER (end) -- write_offset naturally IS both values,
            # just captured at two different points in the loop.
            starts.append(write_offset)
            if n > 0:
                memcpy(dest=dst + write_offset, src=ptr + spans[id].offset, count=n)
                write_offset += n
            ends.append(write_offset)
        return result^

    def token_byte_values(self) -> List[ByteArray]:
        """Return every token's raw bytes as a List, indexed by token ID
        (result[i] == decode_single_token_bytes(i), for all i).

        This materializes a full COPY of every token's bytes at once
        (unlike token_table, which stores them all in one flat shared
        buffer) -- convenient for callers that want a plain
        List[ByteArray] to iterate or index directly, but costs one
        allocation + copy per token. Prefer decode_single_token_bytes()
        for occasional single-token lookups instead of calling this
        repeatedly.

        Returns:
            A list where index i holds token i's raw bytes.
        """
        var result = List[ByteArray](capacity=len(self.token_table))
        var spans = self.token_table.arena.spans.unsafe_ptr()
        var ptr = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
        for i in range(len(self.token_table)):
            var n = spans[i].length
            var bytes = ByteArray(capacity=n)
            bytes.resize(n, 0)
            memcpy(dest=bytes.unsafe_ptr(), src=ptr + spans[i].offset, count=n)
            result.append(bytes^)
        return result^

    def display_of(self, id: Int) raises -> String:
        """Derive the safe-Unicode display string for a token ID from its
        raw bytes via `byte_to_cp` (the bijective byte→codepoint mapping
        set up in __init__).  This is the *display* form used by
        .tiktoken files and encode_single_token; it is derived on demand
        and never stored -- the raw bytes in `token_table` are the single
        source of truth.

        Args:
            id: The token ID to derive the display form for.

        Returns:
            The display string (one codepoint per raw byte).

        Raises:
            Error: if id is negative or >= the vocabulary size.
        """
        if id < 0 or id >= len(self.token_table):
            raise Error("token ID out of range: " + String(id))
        var span = self.token_table.arena.spans[id]
        var raw = self.token_table.arena.bytes.unsafe_ptr()
        var result = String(capacity=span.length * 3)
        for i in range(span.length):
            result += chr(self.byte_to_cp[Int(raw[span.offset + i])])
        return result^

    def encode_single_token[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin]) raises -> Int:
        """Look up the token ID whose DISPLAY string exactly equals
        `text` (an exact single-token match, not general encoding).

        NOT a fast lookup: special tokens are checked via linear scan
        over special_bytes (unavoidable here since the key type is
        String and `text` is a StringSlice, so a direct dict index isn't
        available), and if that fails, the entire vocabulary is scanned
        linearly, deriving each token's display form on demand. Intended
        for occasional lookups / debugging / tooling, not a hot encoding
        path -- use encode() or encode_ordinary() for actual text
        encoding.

        Args:
            text: The exact display string to look up.

        Returns:
            The token ID whose display string equals `text`.

        Raises:
            Error: if no token (special or ordinary) has this exact
                display string.
        """
        for item in self.special_bytes.items():
            if item.key == text:
                return item.value
        for i in range(len(self.token_table)):
            if self.display_of(i) == text:
                return i
        raise Error("unknown token: " + String(text))

    def write_to[T: Writer](self, mut writer: T):
        """Write a short human-readable summary (just the vocab size) to
        `writer` -- backs Writable / print(tokenizer) support."""
        writer.write(
            String("BPETokenizer(vocab_size=")
            + String(len(self.token_table))
            + String(")")
        )

    # ── serialization ────────────────────────────────────────────────────
    # We use the standard .tiktoken format (OpenAI-compatible):
    #   <base64(token_bytes)> <rank>\n
    #
    # See save_tiktoken() / load_tiktoken() below.
    # The legacy JSON-based save()/load() has been removed.
    #
    # A key wrinkle with this format: it stores each token's raw bytes
    # and its rank, but NOT the explicit (a_id, b_id) -> merged_id merge
    # rules this tokenizer needs internally (self.merges /
    # self.lookup_table). load_tiktoken() has to RECONSTRUCT those merge
    # rules purely from the byte strings + ranks, by re-simulating BPE
    # merging on each token's bytes and seeing which two known
    # sub-tokens combine to produce it (_bpe / _recover_merges below).
    # This mirrors the same recovery trick tiktoken's own Python
    # implementation uses when loading a plain .tiktoken file.
    # ─────────────────────────────────────────────────────────────────────

    # ── .tiktoken format support ──────────────────────────────────────

    @staticmethod
    def _bytes_key[
        mut: Bool, //, origin: Origin[mut=mut]
    ](bytes: Span[Byte, origin]) -> String:
        """Build a canonical String key from a raw byte span, e.g. bytes
        [72, 105] -> "72,105". Used as the Dict key for mergeable_ranks
        (byte-sequence -> rank), since raw byte spans themselves can't be
        used directly as Dict keys -- this gives every distinct byte
        sequence a unique, comparable, hashable string representation.

        Args:
            bytes: The raw bytes to encode as a key.

        Returns:
            A comma-separated string of each byte's decimal value.
        """
        var key = String(capacity=len(bytes) * 4)
        for i in range(len(bytes)):
            if i > 0:
                key += ","
            key += String(Int(bytes[i]))
        return key^

    @staticmethod
    def _bpe[
        mut: Bool, //, origin: Origin[mut=mut]
    ](
        mergeable_ranks: Dict[String, Int],
        token_bytes: Span[Byte, origin],
        max_rank: Int,
    ) raises -> List[ByteArray]:
        """Re-simulate BPE merging on `token_bytes`, using only the
        mergeable_ranks table (byte-sequence -> rank) as ground truth,
        and return however many parts it decomposes into once no more
        merges apply.

        This is the core trick behind recovering merge rules from a
        plain .tiktoken file: the file only stores final token byte
        strings and ranks, not the merge tree that produced each one.
        By starting from individual bytes and repeatedly re-applying
        "which adjacent pair, if concatenated, is itself a KNOWN token
        (i.e. present in mergeable_ranks) with the LOWEST rank" -- the
        same greedy-lowest-rank rule real BPE encoding uses -- this
        reconstructs the merge sequence that would have produced
        `token_bytes` in the first place.

        `max_rank` bounds the simulation to only use sub-tokens that were
        "discovered" (had a lower rank, i.e. were learned earlier)
        BEFORE `token_bytes` itself was created as a token -- this
        avoids the simulation "cheating" by using a merge that couldn't
        possibly have existed yet when this specific token was formed
        during original training.

        Args:
            mergeable_ranks: byte-sequence key (see _bytes_key) -> rank,
                built directly from the loaded .tiktoken file.
            token_bytes: The raw bytes of the token being decomposed.
            max_rank: Only concatenations with rank < max_rank are
                considered valid merges (typically the target token's
                own rank/ID, since only earlier-ranked tokens could have
                been its constituent parts).

        Returns:
            The list of byte-sequence parts token_bytes decomposes into
            once no further known-rank merge applies. If merging fully
            succeeds, this typically ends up as 2 parts (the direct
            left/right constituents of the final merge) or occasionally
            more if the merge chain doesn't resolve cleanly to 2.
        """
        # Start with every individual byte as its own separate part.
        var parts = List[ByteArray](capacity=len(token_bytes))
        for i in range(len(token_bytes)):
            var single = ByteArray(capacity=1)
            single.append(token_bytes[i])
            parts.append(single^)

        while True:
            # Scan every ADJACENT pair of current parts, looking for the
            # one whose concatenation is a known token with the lowest
            # rank (i.e. would have been merged EARLIEST during real
            # training) -- same greedy-lowest-rank selection rule used
            # everywhere else in this tokenizer.
            var min_idx = -1
            var min_rank = -1
            for i in range(len(parts) - 1):
                var concat = ByteArray(
                    capacity=len(parts[i]) + len(parts[i + 1])
                )
                for j in range(len(parts[i])):
                    concat.append(parts[i][j])
                for j in range(len(parts[i + 1])):
                    concat.append(parts[i + 1][j])
                var key = Self._bytes_key(Span[Byte](concat))
                if key in mergeable_ranks:
                    var rank = mergeable_ranks[key]
                    if min_idx < 0 or rank < min_rank:
                        min_idx = i
                        min_rank = rank
            # Stop if no adjacent pair concatenates into any known token,
            # OR if the best candidate's rank isn't actually earlier than
            # max_rank (meaning it couldn't have been a real constituent
            # of the token we're decomposing).
            if min_idx < 0 or (max_rank >= 0 and min_rank >= max_rank):
                break

            # Apply the merge: replace parts[min_idx] and
            # parts[min_idx+1] with their single concatenated form,
            # rebuilding the parts list around that change.
            var merged = ByteArray(
                capacity=len(parts[min_idx]) + len(parts[min_idx + 1])
            )
            for j in range(len(parts[min_idx])):
                merged.append(parts[min_idx][j])
            for j in range(len(parts[min_idx + 1])):
                merged.append(parts[min_idx + 1][j])
            var new_parts = List[ByteArray](capacity=len(parts) - 1)
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
        all_tokens: List[ByteArray],
    ) raises:
        """Reconstruct self.merges (the (a_id, b_id, merged_id) rule
        list) purely from the loaded .tiktoken file's token bytes and
        ranks, since the file itself doesn't store merge rules directly.

        For every token beyond the base 256 bytes, decompose its bytes
        via _bpe() and figure out which TWO existing token IDs combined
        to form it -- either directly (if _bpe reduces it to exactly 2
        parts, both of which are themselves known tokens), or by
        searching for the best splitting point among more than 2 parts
        (if the straightforward 2-part decomposition doesn't apply
        cleanly). This mirrors tiktoken's own reference approach for
        recovering merges from a plain-format vocabulary file.

        Args:
            mergeable_ranks: byte-sequence key -> rank, from the loaded
                file.
            all_tokens: token_id -> raw bytes, from the loaded file
                (index == rank == token ID).
        """
        var size = len(all_tokens)
        var recovered = List[MergeRule]()
        for token_id in range(256, size):
            var token_bytes = all_tokens[token_id].copy()
            var n = len(token_bytes)
            if n <= 1:
                # Single-byte or empty tokens can't be the result of a
                # merge (a merge always combines two non-empty parts).
                continue
            var parts = self._bpe(
                mergeable_ranks, Span[Byte](token_bytes), token_id
            )
            var left_id = -1
            var right_id = -1

            if len(parts) == 2:
                # Clean case: _bpe's simulation reduced this token to
                # exactly two parts -- if BOTH are themselves known
                # tokens, they're almost certainly the direct left/right
                # constituents of the merge that created token_id.
                var lk = Self._bytes_key(Span[Byte](parts[0]))
                var rk = Self._bytes_key(Span[Byte](parts[1]))
                if lk in mergeable_ranks and rk in mergeable_ranks:
                    left_id = mergeable_ranks[lk]
                    right_id = mergeable_ranks[rk]
            elif len(parts) > 2:
                # Messier case: simulation didn't reduce cleanly to 2
                # parts. Search every adjacent pair among the remaining
                # parts for the best candidate split point -- "best"
                # meaning: its concatenation IS a known token, that
                # token's rank is below token_id (so it could have
                # existed already), and it's the HIGHEST such rank found
                # so far (closest in learn-order to token_id itself,
                # i.e. the most recently-learned valid candidate --
                # picking the highest qualifying rank rather than the
                # first one found reduces the chance of picking an
                # unrelated coincidental match).
                var best_cr = -1
                for i in range(len(parts) - 1):
                    var concat = ByteArray(
                        capacity=len(parts[i]) + len(parts[i + 1])
                    )
                    for k in range(len(parts[i])):
                        concat.append(parts[i][k])
                    for k in range(len(parts[i + 1])):
                        concat.append(parts[i + 1][k])
                    var ck = Self._bytes_key(Span[Byte](concat))
                    if ck in mergeable_ranks:
                        var cr = mergeable_ranks[ck]
                        if cr < token_id and cr > best_cr:
                            var lk2 = Self._bytes_key(Span[Byte](parts[i]))
                            var rk2 = Self._bytes_key(Span[Byte](parts[i + 1]))
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
                # Successfully identified both constituent IDs -- record
                # the reconstructed merge rule. (If neither branch above
                # found a valid pair, this token_id is silently skipped
                # -- its merge rule couldn't be recovered, which can
                # happen for tokens whose formation doesn't decompose
                # cleanly via this greedy simulation.)
                recovered.append(MergeRule(left_id, right_id, token_id))
        self.merges = recovered^

    def save_tiktoken(mut self, path: String) raises:
        """Write this tokenizer's vocabulary to `path` in the standard
        .tiktoken plaintext format: one line per token, formatted as
        "<base64(raw_bytes)> <rank>\\n", ordered by token ID.

        Special tokens are excluded (per the .tiktoken convention --
        they're typically distributed/configured separately from the
        base mergeable vocabulary), as are any empty placeholder slots
        left over from special-token ID gaps (see
        _register_special_token's set_bytes gap-padding behavior).

        Args:
            path: File path to write the .tiktoken file to.
        """
        var spans = self.token_table.arena.spans.unsafe_ptr()
        var pool = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
        with open(path, "w") as f:
            for token_id in range(len(self.token_table)):
                if token_id in self.inverse_special:
                    continue
                # Write the token's actual RAW bytes (the .tiktoken format
                # stores real byte sequences) directly from the span
                # table -- since token_table holds raw bytes verbatim,
                # the display string and the stored bytes coincide.
                var span = spans[token_id]
                if span.length == 0:
                    continue
                var raw = ByteArray(capacity=span.length)
                raw.resize(span.length, 0)
                memcpy(
                    dest=raw.unsafe_ptr(),
                    src=pool + span.offset,
                    count=span.length,
                )
                var encoded = b64encode(Span[Byte](raw))
                f.write(encoded + " " + String(token_id) + "\n")

    def load_tiktoken(mut self, path: String) raises:
        """Load a vocabulary from a .tiktoken-format file at `path`,
        replacing this tokenizer's current token_table/merges/
        lookup_table.

        Since the .tiktoken format only stores (raw_bytes, rank) pairs
        and NOT explicit merge rules, this reconstructs everything else
        the tokenizer needs: stores the raw bytes verbatim in
        token_table, recovers merge rules via _recover_merges
        (re-simulating BPE per-token), rebuilds lookup_table from those
        recovered rules, and repopulates byte_to_rank so single-byte
        encoding also respects the loaded file's rank assignments.

        Args:
            path: File path to load the .tiktoken file from.
        """
        var file_content: String
        with open(path, "r") as f:
            file_content = f.read()
        var raw_lines = file_content.split("\n")

        # ---- Parse every line into (raw_bytes, rank) pairs -------------
        var mergeable_ranks = Dict[String, Int]()  # byte-sequence key -> rank
        var all_tokens = List[ByteArray]()  # rank (== token id) -> raw bytes
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
            # Pad all_tokens with empty placeholders up to `rank`, same
            # gap-filling approach as _register_special_token, since
            # ranks in the file may not be strictly sequential/contiguous
            # as they're parsed (though they typically are in practice).
            while len(all_tokens) <= rank:
                all_tokens.append(ByteArray())
            all_tokens[rank] = raw^
            if rank > max_id:
                max_id = rank

        # ---- Rebuild token_table (raw bytes) -----------------------------
        # Store each token's raw bytes VERBATIM -- no display-string
        # round-trip.  (The old code re-derived the safe-Unicode display
        # string and converted it back, which could silently corrupt
        # tokens whose raw bytes collide with the GPre spacer sequence
        # 0xC4 0xA0, e.g. o200k's " Ġ".  Since display is now derived on
        # demand from the raw bytes, the raw bytes are the only thing
        # that needs to be stored.)
        var new_vocab_size = max_id + 1
        var new_table = TokenByteTable()
        new_table.reserve(new_vocab_size)
        for token_id in range(new_vocab_size):
            new_table.add(Span[Byte](all_tokens[token_id]))

        # ---- Recover merge rules and rebuild the O(1) lookup table -----
        self._recover_merges(mergeable_ranks, all_tokens)

        var new_lookup_table = MergeLookup()
        for merge in self.merges:
            new_lookup_table.set(merge.first, merge.second, merge.merged)

        # Populate byte_to_rank from the loaded file's rank assignments.
        # For SEQUENTIAL this usually differs from identity; for SHUFFLED
        # it matches the comptime LUT so the change is a no-op.
        var single_byte = ByteArray()
        single_byte.resize(1, 0)
        for b in range(256):
            single_byte[0] = Byte(b)
            var key = BPETokenizer._bytes_key(
                Span[Byte](ptr=single_byte.unsafe_ptr(), length=1)
            )
            if key in mergeable_ranks:
                self.byte_to_rank[b] = mergeable_ranks[key]

        # ---- Commit the rebuilt state ----------------------------------
        self.token_table = new_table^
        self.lookup_table = new_lookup_table^

        # Re-register any special tokens defined by the pre-tokenizer's
        # own default set that weren't already present in the loaded
        # file (the loaded file's plain vocab may not include special
        # tokens, per save_tiktoken's exclusion of them on write).
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

    Used by encode_ordinary()'s short-word path: applies ONE pass of
    merging the specific pair (a, b) -> m everywhere it occurs in
    buf[0:n], compacting the buffer in place (same read-cursor/
    write-cursor technique used in WordCounts and the training merge
    loop elsewhere in this file). The caller is responsible for calling
    this repeatedly, once per merge, since this only applies a single
    specific pair per call -- it does not search for the best pair
    itself (that's done by the caller before each call).
    """
    var w = 0
    var i = 0
    while i < n:
        if i < n - 1 and buf[i] == a and buf[i + 1] == b:
            # Match found: collapse both source tokens into the single
            # merged id, advance the read cursor past both.
            buf[w] = m
            i += 2
        else:
            # No match here: copy the token forward unchanged (position
            # may still shift left due to compaction from an earlier
            # match in this same pass).
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
    def train[
        mut: Bool, //, origin: Origin[mut=mut], T: PreTokenizer
    ](corpus: Span[String, origin], vocab_size: Int) raises -> BPETokenizer[T]:
        var tok = BPETokenizer[T]()
        tok.train(corpus, vocab_size)
        return tok^
