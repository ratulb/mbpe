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
from std.memory import alloc, free
from std.atomic import Atomic, Ordering, fence
from std.sys import size_of


# ---------------------------------------------------------------------------
# Pre-tokeniser — approximates GPT-2's whitespace handling.
#
# GPT-2 uses a regex to split on category boundaries (letters vs numbers vs
# punctuation vs whitespace) so that BPE never merges across them.  Our
# version is simpler: it replaces each space with " <spacer_byte>" before
# splitting on spaces, then reinserts the spacer on the front of each word.
# This approximates the `Ġ` convention used by GPT-2 / tiktoken where a
# leading Ġ means "this word was preceded by a space".
#
# The spacer character (U+0120, Latin capital letter G with inverted breve)
# was chosen by OpenAI because it almost never appears in real text.
#
# Accepts StringSlice (any string view) to avoid forcing callers to own a
# String.  Returns owned Strings because the words need to outlive the
# input text (they become dict keys in the caller).
# ---------------------------------------------------------------------------

struct PreTokenizer:
    @staticmethod
    def tokenize[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
        spacer: StaticString = "Ġ",
    ](text: StringSlice[origin]) raises -> List[String]:
        var splits = (
            text
            .replace(" ", " " + spacer)
            .replace(".", " .")
            .split(" ")
        )
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


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
#   merges      : ordered merge rules         (List[Tuple[Int, Int, Int]])
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

struct BPETokenizer(Sized & Movable):
    var vocab: List[String]
    var merges: List[Tuple[Int, Int, Int]]
    var merge_cache: PairCache
    var byte_to_cp: Dict[Int, Int]
    var cp_to_byte: Dict[Int, Int]

    def __init__(out self):
        self.vocab = List[String]()
        self.merges = List[Tuple[Int, Int, Int]]()
        self.merge_cache = PairCache()
        self.byte_to_cp = Dict[Int, Int]()
        self.cp_to_byte = Dict[Int, Int]()

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
            var words = PreTokenizer.tokenize(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # ---- 2. Build byte ↔ safe-Unicode mapping -----------------------
        self.byte_to_cp = Dict[Int, Int]()
        self.cp_to_byte = Dict[Int, Int]()
        var n = 0
        for b in range(256):
            # Printable ranges as defined by GPT-2's encoder.py.
            # Everything else (control chars, whitespace, DEL, soft hyphen)
            # gets a new codepoint ≥ 256.
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

        # ---- 3. Initialise vocabulary -----------------------------------
        # IDs 0–255 are the 256 byte values, each mapped to its safe-Unicode
        # representation.  Merge tokens are appended below.
        self.vocab = List[String](capacity=vocab_size)
        for b in range(256):
            self.vocab.append(chr(self.byte_to_cp[b]))

        # ---- 4. Split each word into base token IDs ---------------------
        # Every word is a sequence of raw UTF-8 bytes.  Each byte value is
        # its own token ID (0x00 → 0, 0x01 → 1, ..., 0xFF → 255).
        var splits = Dict[String, List[Int]]()
        for word in word_freqs.keys():
            var sb = word.as_bytes()
            var ids = List[Int](capacity=len(sb))
            for i in range(len(sb)):
                ids.append(Int(sb[i]))
            splits[word] = ids^

        # ---- 5. Merge loop ----------------------------------------------
        # Each iteration finds the most frequent adjacent pair across all
        # words and replaces every occurrence with a single new token.
        self.merges = List[Tuple[Int, Int, Int]]()
        while len(self.vocab) < vocab_size:
            # Count how often each pair of adjacent token-IDs appears.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            if len(pair_freqs) == 0:
                # No pairs left to merge — every word is a single token.
                break

            # Pick the pair with the highest frequency.
            var best_pair: Tuple[Int, Int] = (0, 0)
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            # Apply the merge and record it.
            var merged_id = len(self.vocab)
            var a_id = best_pair[0]
            var b_id = best_pair[1]
            _merge_pair(a_id, b_id, merged_id, splits, word_freqs)
            # Store in order so encoding always applies merges in the same
            # sequence they were learned.
            self.merges.append((a_id, b_id, merged_id))
            self.merge_cache.set(a_id, b_id, merged_id)
            # The display string is the concatenation of the two parts.
            # We copy to avoid aliasing: self.vocab[a_id] is a view into
            # the list, and append might reallocate.
            var merged_str = self.vocab[a_id].copy() + self.vocab[b_id].copy()
            self.vocab.append(merged_str)

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
        var words = PreTokenizer.tokenize(text)
        # ---- 2. Per word: bytes → Ints, greedy rank-based merge ---------
        var result = List[Int]()
        for word in words:
            var ptr = word.unsafe_ptr()
            var n = word.byte_length()
            var start = len(result)

            # Reserve space in result (one bulk extension)
            result.resize(start + n, 0)
            var dst = result.unsafe_ptr() + start

            # Copy bytes as Ints into result's tail
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

            # Trim excess from merge shrinkage
            while len(result) > start + n:
                _ = result.pop()

        return result^

    def encode[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> List[Int]:
        return self._tokenize(text)

    # ── decoding ─────────────────────────────────────────────────────────
    # Decode is the inverse of encode:
    #   1. Look up each token ID in vocab → safe-Unicode display string.
    #   2. Reverse-map every safe-Unicode codepoint back to its raw byte.
    #   3. Interpret the byte sequence as UTF-8 (lossy — invalid bytes
    #      become the U+FFFD replacement character).
    #   4. Replace the pre-tokeniser's Ġ spacer with regular spaces.
    # ─────────────────────────────────────────────────────────────────────

    def decode[mut: Bool, //, origin: Origin[mut=mut]](self, ids: Span[Int, origin]) raises -> String:
        if len(ids) == 0:
            return String("")
        # ---- 1. Reverse-map token display codepoints → raw bytes --------
        var byte_list = List[UInt8](capacity=len(ids))
        for id in ids:
            for cp in self.vocab[id].codepoints():
                byte_list.append(UInt8(self.cp_to_byte[Int(cp)]))
        # ---- 2. Interpret bytes as UTF-8 --------------------------------
        var decoded = String(from_utf8=Span[UInt8](byte_list))
        # ---- 3. Restore spaces from the Ġ convention --------------------
        return StringSlice(decoded).replace("Ġ", " ")

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
        for (a_id, b_id, merged_id) in self.merges:
            var entry = Python.list()
            entry.append(Python.int(a_id))
            entry.append(Python.int(b_id))
            entry.append(Python.int(merged_id))
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
        for i in range(len(py_vocab)):
            tok.vocab.append(String(py_vocab[i]))

        # Rebuild ordered merge list.
        var py_merges = data["merges"]
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            var a_id = Int(py=entry[0])
            var b_id = Int(py=entry[1])
            var merged_id = Int(py=entry[2])
            tok.merges.append((a_id, b_id, merged_id))
            tok.merge_cache.set(a_id, b_id, merged_id)

        return tok^


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


# ═══════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════
#
# These are separate functions rather than methods because they take mutable
# references to the splits dictionary and don't need access to the tokenizer
# struct's fields.

def _compute_pair_freqs(
    splits: Dict[String, List[Int]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[Int, Int], Int]:
    """Count how often each adjacent pair of token-IDs appears in the corpus.

    For each word, look at its current token-ID split and tally every pair
    of adjacent IDs weighted by the word's frequency.  Words that are already
    a single token are skipped (no pairs to count).
    """
    var pair_freqs = Dict[Tuple[Int, Int], Int]()
    for word_freq in word_freqs.items():
        ref word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a_id: Int,
    b_id: Int,
    merged_id: Int,
    mut splits: Dict[String, List[Int]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every occurrence of (a_id, b_id) with merged_id.

    For every word in the corpus, scan its token-ID list.  When the pair
    (a_id, b_id) is found, replace the pair with merged_id and do NOT
    advance the scan position — the newly inserted token might itself be
    the left half of another match at the same position.

    This function mutates `splits` in place.  The final `.copy()` ensures
    the dict entry is an owned value (the intermediate list comprehensions
    may create references).
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a_id and split[i + 1] == b_id:
                # Rebuild the split: elements before the pair + merged
                # token + elements after the pair.  i stays the same so
                # we can immediately check (merged_id, split[i+1]).
                split = (
                    [e for e in split[:i]]
                    + [merged_id]
                    + [e for e in split[i + 2:]]
                )
            else:
                i += 1
        splits[word] = split.copy()
