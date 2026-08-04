"""Pure-Mojo pre-tokenizers for BPE tokenizer training and encoding.

================================================================================
WHAT IS PRE-TOKENIZATION?
================================================================================

Before BPE merge rules are applied (during both training and encoding), the raw
input text must be split into "words" -- the fundamental units the BPE algorithm
operates on.  This splitting step is called pre-tokenization.

Different tokenizer families use different pre-tokenization strategies:

  - GPT-2 / r50k_base -- a hand-written regex with 7 alternatives.  Spaces
    are kept inline with the following token (no special spacer character).
    Decode is a simple UTF-8 byte concatenation.

  - GPT-4 / cl100k_base -- a similar but distinct 8-alternative regex.
    Digit runs are capped at 3 codepoints; letter runs can have a non-letter
    prefix; newlines are an explicit alternative.

  - GPT-4 / cl100k_base -- a similar but distinct 8-alternative regex.
    Digit runs are capped at 3 codepoints; letter runs can have a non-letter
    prefix; newlines are an explicit alternative.

================================================================================
ARCHITECTURE
================================================================================

                       +------------------------------+
                       |      BPETokenizer[PT]        |
                       |  (tokenizer.mojo)            |
                       |                              |
                       |  PT.split(text) -> words     |  <- calls into
                       |  for each word:              |     the trait
                       |    encode bytes -> merge     |
                       +------------------------------+
                                    |
                                    | uses
                                    v
                       +------------------------------+
                       |    PreTokenizer (trait)      |
                       |                              |
                       |  split(text: String)         |
                       |    -> List[String]           |
                       +------------------------------+
                          ^              ^
                          |              |
               +----------+---------+  +-+--------------+
               |GPT2PreTok.        |  |GPT4PreTok.     |
               |(r50k_base)        |  |(cl100k_base)   |
               +-------------------+  +----------------+

Each struct implements the PreTokenizer trait and provides a split() method.
The choice of which implementation to use is selected at compile time via the
PT parameter of BPETokenizer[PT: PreTokenizer = GPT2Pretokenizer].

================================================================================
HOW IT PLUGS INTO BPETokenizer
================================================================================

During training (train()):
  1. BPETokenizer calls self.pt.split(text) for each training line
  2. Each resulting word string is converted to byte-token IDs (0-255)
  3. Pair frequencies are tallied and merge rules learned on those IDs

During encoding (encode() / _tokenize()):
  1. BPETokenizer calls self.pt.split(text) on the input
  2. Each word string is converted to byte-token IDs
  3. Merge rules are applied greedily (lowest-merged_id pair first)
  4. The resulting token-ID sequence is returned

The pre-tokenizer thus determines which byte sequences become BPE "words" and
therefore what boundaries the merge rules can never cross.

================================================================================
"FIRST MATCH WINS" ALTERNATION SEMANTICS
================================================================================

The regex patterns use alternation (|), where alternatives are tried
left-to-right and the first one that matches at the current position
wins -- NOT the longest match.  This is standard regex alternation semantics.

Our _best_match() methods implement exactly this: each matcher is tried in
order, and the first one that returns a positive match length is accepted.

================================================================================
FILE STRUCTURE
================================================================================

  Section                              | Description
  -------------------------------------+---------------------------------
  Shared UTF-8 helpers                 | Low-level byte/codepoint utils
  PreTokenizer trait                   | Interface that all impls satisfy
  GPT2Pretokenizer (r50k_base)         | 7-matcher regex (GPT-2)
  GPT4Pretokenizer (cl100k_base)       | 8-matcher regex (GPT-4)
"""

# ===========================================================================
# Shared UTF-8 helpers
# ===========================================================================
#
# All pre-tokenizers operate on raw UTF-8 byte spans
# (Span[UInt8]).  These helpers abstract away the UTF-8 decoding so
# that the matchers can work at the codepoint level.
#
# The helpers are marked @always_inline because they're called from
# hot loops -- inlining eliminates function-call overhead.

from std.bit import pop_count
from std.memory import memcmp


@always_inline
def utf8_byte_length(lead: UInt8) -> Int:
    """Return the byte-length of a UTF-8 codepoint given its leading byte.

    UTF-8 encoding:
      - 1 byte:  0xxxxxxx  (lead byte < 0x80)
      - 2 bytes: 110xxxxx  (lead byte < 0xE0)
      - 3 bytes: 1110xxxx  (lead byte < 0xF0)
      - 4 bytes: 11110xxx  (lead byte >= 0xF0)

    Args:
        lead: The leading byte of a UTF-8 sequence.

    Returns:
        1, 2, 3, or 4 -- the number of bytes in this codepoint.
    """
    if lead < 0x80:
        return 1
    if lead < 0xE0:
        return 2
    if lead < 0xF0:
        return 3
    return 4


@always_inline
def decode_codepoint[
    origin: Origin, //
](ptr: UnsafePointer[UInt8, origin], length: Int) -> Int:
    """Decode a single Unicode codepoint from raw UTF-8 bytes.

    Reads `length` bytes from `ptr` and reconstructs the codepoint.
    No validation is performed (assumes valid UTF-8).

    Args:
        ptr: Pointer to the first byte of a UTF-8 encoding.
        length: Number of bytes (1-4, from utf8_byte_length).

    Returns:
        The decoded Unicode codepoint as an integer.
    """
    var b0 = Int(ptr.load(0))
    if length == 1:
        return b0
    var b1 = Int(ptr.load(1))
    if length == 2:
        return ((b0 & 0x1F) << 6) | (b1 & 0x3F)
    var b2 = Int(ptr.load(2))
    if length == 3:
        return ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
    var b3 = Int(ptr.load(3))
    return (
        ((b0 & 0x07) << 18)
        | ((b1 & 0x3F) << 12)
        | ((b2 & 0x3F) << 6)
        | (b3 & 0x3F)
    )


@always_inline
def is_letter(cp: Int) -> Bool:
    """Return True if cp is a Unicode letter (like \\p{L} in regex)."""
    if cp < 128:
        return (65 <= cp <= 90) or (97 <= cp <= 122)
    return (
        (0x00AA <= cp and cp <= 0x00AA)
        or (0x00B5 <= cp and cp <= 0x00B5)
        or (0x00BA <= cp and cp <= 0x00BA)
        or (0x00C0 <= cp and cp <= 0x00D6)
        or (0x00D8 <= cp and cp <= 0x00F6)
        or (0x00F8 <= cp and cp <= 0x024F)
        or (0x0250 <= cp and cp <= 0x02AF)
        or (0x0300 <= cp and cp <= 0x036F)
        or (0x1E00 <= cp and cp <= 0x1EFF)
        or (0x0400 <= cp and cp <= 0x04FF)
        or (0x0500 <= cp and cp <= 0x052F)
        or (0x0370 <= cp and cp <= 0x03FF)
        or (0x1F00 <= cp and cp <= 0x1FFF)
        or (0x2E80 <= cp and cp <= 0x2EFF)
        or (0x2F00 <= cp and cp <= 0x2FDF)
        or (0x3000 <= cp and cp <= 0x303F)
        or (0x3040 <= cp and cp <= 0x309F)
        or (0x30A0 <= cp and cp <= 0x30FF)
        or (0x3100 <= cp and cp <= 0x312F)
        or (0x3130 <= cp and cp <= 0x318F)
        or (0x3200 <= cp and cp <= 0x32FF)
        or (0x3300 <= cp and cp <= 0x33FF)
        or (0x3400 <= cp and cp <= 0x4DBF)
        or (0x4E00 <= cp and cp <= 0x9FFF)
        or (0xA000 <= cp and cp <= 0xA4CF)
        or (0xAC00 <= cp and cp <= 0xD7AF)
        or (0xF900 <= cp and cp <= 0xFAFF)
        or (0xFE30 <= cp and cp <= 0xFE4F)
        or (0xFF21 <= cp and cp <= 0xFF3A)
        or (0xFF41 <= cp and cp <= 0xFF5A)
        or (0x0600 <= cp and cp <= 0x06FF)
        or (0x0750 <= cp and cp <= 0x077F)
        or (0x08A0 <= cp and cp <= 0x08FF)
        or (0x0590 <= cp and cp <= 0x05FF)
        or (0x0900 <= cp and cp <= 0x097F)
        or (0x0980 <= cp and cp <= 0x09FF)
        or (0x0E00 <= cp and cp <= 0x0E7F)
    )


@always_inline
def is_digit(cp: Int) -> Bool:
    """Return True if cp is a Unicode digit (like \\p{N} in regex)."""
    if cp < 128:
        return 48 <= cp <= 57
    return (
        (0x0030 <= cp and cp <= 0x0039)
        or (0x0660 <= cp and cp <= 0x0669)
        or (0x06F0 <= cp and cp <= 0x06F9)
        or (0x0966 <= cp and cp <= 0x096F)
        or (0x09E6 <= cp and cp <= 0x09EF)
        or (0x0AE6 <= cp and cp <= 0x0AEF)
        or (0x0B66 <= cp and cp <= 0x0B6F)
        or (0x0BE6 <= cp and cp <= 0x0BEF)
        or (0x0C66 <= cp and cp <= 0x0C6F)
        or (0x0CE6 <= cp and cp <= 0x0CEF)
        or (0x0D66 <= cp and cp <= 0x0D6F)
        or (0x0E50 <= cp and cp <= 0x0E59)
        or (0x0ED0 <= cp and cp <= 0x0ED9)
        or (0x0F20 <= cp and cp <= 0x0F29)
    )


@always_inline
def is_letter_or_digit(cp: Int) -> Bool:
    """Return True if cp is a letter or digit.

    Equivalent to is_letter(cp) or is_digit(cp).
    Used in character classes like [^...\\p{L}\\p{N}] in the regex patterns.
    """
    return is_letter(cp) or is_digit(cp)


@always_inline
def is_lowercase(cp: Int) -> Bool:
    """Return True if cp is a Unicode lowercase letter (like \\p{Ll} in regex).
    """
    if cp < 128:
        return 97 <= cp <= 122
    # Latin-1 Supplement lowercase
    if 0x00AA == cp or 0x00B5 == cp or 0x00BA == cp:
        return True
    if 0x00DF <= cp <= 0x00F6:
        return True
    if 0x00F8 <= cp <= 0x00FF:
        return True
    # Latin Extended: odd codepoints = lowercase, even = uppercase
    if 0x0100 <= cp <= 0x024F:
        return (cp & 1) == 1
    # IPA Extensions (all Ll)
    if 0x0250 <= cp <= 0x02AF:
        return True
    # Greek: 0x03B1-0x03C9 alpha-omega lowercase, 0x03CE, 0x03D0-0x03E1, etc.
    if 0x03B1 <= cp <= 0x03C9:
        return True
    if cp == 0x03CE:
        return True
    # Cyrillic lowercase: 0x0430-0x044F + extended
    if 0x0430 <= cp <= 0x044F:
        return True
    if 0x0450 <= cp <= 0x04FF and ((cp - 0x0450) % 2 == 0 or cp == 0x0450):
        return True
    # Latin Extended Additional: odd = lowercase
    if 0x1E00 <= cp <= 0x1EFF:
        return (cp & 1) == 1
    # Greek Extended: 0x1F00-0x1FFF — many lowercase forms
    if 0x1F00 <= cp <= 0x1FFF:
        return True
    # Fullwidth a-z
    if 0xFF41 <= cp <= 0xFF5A:
        return True
    return False


@always_inline
def is_mark(cp: Int) -> Bool:
    """Return True if cp is a combining mark (like \\p{M} in regex)."""
    if cp < 128:
        return False
    return (
        (0x0300 <= cp and cp <= 0x036F)  # Combining Diacritical Marks
        or (0x0483 <= cp and cp <= 0x0489)  # Cyrillic
        or (0x0591 <= cp and cp <= 0x05BD)  # Hebrew
        or cp == 0x05BF
        or cp == 0x05C1
        or cp == 0x05C2
        or cp == 0x05C4
        or cp == 0x05C5
        or cp == 0x05C7
        or (0x0610 <= cp and cp <= 0x061A)  # Arabic
        or (0x064B <= cp and cp <= 0x065F)
        or cp == 0x0670
        or (0x06D6 <= cp and cp <= 0x06DC)
        or (0x06DF <= cp and cp <= 0x06E4)
        or (0x06E7 <= cp and cp <= 0x06E8)
        or (0x06EA <= cp and cp <= 0x06ED)
        or (0x0711 <= cp and cp <= 0x074A)  # Syriac
        or (0x0901 <= cp and cp <= 0x0903)  # Devanagari
        or cp == 0x093C
        or (0x093E <= cp and cp <= 0x094D)
        or (0x0951 <= cp and cp <= 0x0954)
        or (0x0962 <= cp and cp <= 0x0963)
        or (0x0981 <= cp and cp <= 0x0983)  # Bengali
        or cp == 0x09BC
        or (0x09BE <= cp and cp <= 0x09C4)
        or cp == 0x09C7
        or cp == 0x09C8
        or (0x09CB <= cp and cp <= 0x09CD)
        or cp == 0x09D7
        or (0x09E2 <= cp and cp <= 0x09E3)
        or (0x0E31 <= cp and cp <= 0x0E3A)  # Thai
        or (0x0E47 <= cp and cp <= 0x0E4E)
        or (0x0F71 <= cp and cp <= 0x0F84)  # Tibetan
        or (0x0F86 <= cp and cp <= 0x0F87)
        or (0x0F90 <= cp and cp <= 0x0FBC)
        or (0x102B <= cp and cp <= 0x103E)  # Myanmar
        or (0x17B6 <= cp and cp <= 0x17D3)  # Khmer
        or (
            0x1DC0 <= cp and cp <= 0x1DFF
        )  # Combining Diacritical Marks Supplement
        or (0x20D0 <= cp and cp <= 0x20F0)  # Combining Marks for Symbols
        or (0xFE00 <= cp and cp <= 0xFE0F)  # Variation Selectors
        or (0xFE20 <= cp and cp <= 0xFE2F)  # Combining Half Marks
        or cp == 0x200C
        or cp == 0x200D  # ZWJ/ZWNJ
    )


@always_inline
def is_upper_like(cp: Int) -> Bool:
    """Like \\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M} for o200k (not-Ll letters + marks).

    Note: \\p{N} (digits) are excluded because o200k's alt 1/2 letter
    patterns don't include the \\p{N} category.  The ``is_letter`` helper
    includes digit codepoints that happen to share a Unicode block with
    letters (e.g. Thai 0x0E50-0x0E59), so we must explicitly prune them.
    """
    if cp < 128:
        return 65 <= cp <= 90
    return (
        is_letter(cp) and not is_lowercase(cp) and not is_digit(cp)
    ) or is_mark(cp)


@always_inline
def is_lower_like(cp: Int) -> Bool:
    """Like \\p{Ll}\\p{Lm}\\p{Lo}\\p{M} for o200k (Ll + other letters + marks).

    Same digit-exclusion rationale as ``is_upper_like``.
    """
    if cp < 128:
        return (65 <= cp <= 90) or (97 <= cp <= 122)
    return (is_letter(cp) and not is_digit(cp)) or is_mark(cp)


@always_inline
def is_whitespace(cp: Int) -> Bool:
    """Return True if cp is a Unicode whitespace codepoint."""
    if cp < 128:
        return cp == 0x0009 or cp == 0x000A or cp == 0x000D or cp == 0x0020
    return (
        cp == 0x0009
        or cp == 0x000A
        or cp == 0x000B
        or cp == 0x000C
        or cp == 0x000D
        or cp == 0x0020
        or cp == 0x0085
        or cp == 0x00A0
        or cp == 0x1680
        or cp == 0x2000
        or cp == 0x2001
        or cp == 0x2002
        or cp == 0x2003
        or cp == 0x2004
        or cp == 0x2005
        or cp == 0x2006
        or cp == 0x2007
        or cp == 0x2008
        or cp == 0x2009
        or cp == 0x200A
        or cp == 0x2028
        or cp == 0x2029
        or cp == 0x202F
        or cp == 0x205F
        or cp == 0x3000
    )


# ── ASCII byte-class LUT ─────────────────────────────────────────────────
# Bitmask per byte value, used by the matchers' ASCII fast paths to avoid
# UTF-8 decode + Unicode range checks for the ASCII range (0x00-0x7F).
# Non-ASCII lead bytes (>= 0x80) have class 0 and fall back to the
# codepoint-decoding matcher paths.  This is the hot-table for both
# training and encode pre-tokenization.
comptime BC_LETTER = 1  # ASCII \p{L}   (A-Z, a-z)
comptime BC_DIGIT = 2  # ASCII \p{N}   (0-9)
comptime BC_WS = 4  # ASCII \s      (0x09-0x0D, 0x20)
comptime BC_CRLF = 8  # CR or LF      (0x0A, 0x0D)

comptime BYTE_CLASS = SIMD[DType.int32, 256](
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 4, 12, 4, 4, 12, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
)



# ── SWAR helpers (ASCII class scan, 8 bytes at a time) ────────────────────
# These implement the same byte classes as BYTE_CLASS but operate on 8
# bytes packed into a UInt64, so the letter/digit runs of the ASCII fast
# paths can be scanned 8x faster than the per-byte LUT loop.  Non-ASCII
# bytes (>= 0x80) never test as class members, so the SWAR loops stop
# exactly at the first byte that needs the codepoint-decoding path.
@always_inline
def _hasless(x: UInt64, n: UInt64) -> UInt64:
    return (
        (x - n * UInt64(0x0101010101010101)) & ~x & UInt64(0x8080808080808080)
    )


@always_inline
def _letters8(x: UInt64) -> UInt64:
    var l = x | UInt64(0x2020202020202020)
    var below_z = _hasless(l, UInt64(0x7B))
    var below_a = _hasless(l, UInt64(0x61))
    return below_z & ~below_a


@always_inline
def _ctpop64(x: UInt64) -> Int:
    var y = x - ((x >> 1) & UInt64(0x5555555555555555))
    y = (y & UInt64(0x3333333333333333)) + (
        (y >> 2) & UInt64(0x3333333333333333)
    )
    y = (y + (y >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
    return Int((y * UInt64(0x0101010101010101)) >> 56)


@always_inline
def _swar_letter_run[
    origin: Origin, //
](span: Span[UInt8, origin], i: Int, n: Int) -> Int:
    """Consume consecutive ASCII letters starting at i, 8 bytes at a time.

    Returns the number of bytes consumed.  Stops at the first non-letter
    byte, the first non-ASCII byte (>= 0x80), or end-of-span — matching
    the semantics of the per-byte \\p{L} scan, just faster.  The caller
    continues with the codepoint-decoding path for any non-ASCII byte
    that stopped the scan.
    """
    var p8 = span.unsafe_ptr()
    var consumed = 0
    var j = i
    while j + 8 <= n:
        var w: UInt64 = (p8 + j).bitcast[UInt64]()[]
        var nl = ~_letters8(w) & UInt64(0x8080808080808080)
        if nl != 0:
            var lsb = nl & (UInt64(0) - nl)
            # return consumed + (_ctpop64(lsb - 1) >> 3)
            return consumed + (pop_count(lsb - 1).__int__() >> 3)
        consumed += 8
        j += 8
    while j < n:
        var b = span[j]
        if b < 0x80 and Int(BYTE_CLASS[Int(b)]) & BC_LETTER:
            consumed += 1
            j += 1
        else:
            break
    return consumed


@always_inline
def is_ascii_ws_byte(b: Int) -> Bool:
    """Return True if byte b is an ASCII whitespace byte.

    Matches: \\t (0x09), \\n (0x0A), \\v (0x0B), \\f (0x0C),
    \\r (0x0D), space (0x20).

    This is the byte-level version of \\s from the regex patterns.
    We use byte checks (not codepoint checks) because:
      - GPT-2/GPT-4 regex \\s matches ONLY ASCII whitespace bytes
      - The match_ws_* and _match_newline matchers operate on raw
        bytes, not decoded codepoints
      - Byte-level comparisons are faster and avoid UTF-8 decoding
        overhead

    This helper exists because the same ASCII-whitespace check is used
    by multiple matchers in both GPT2Pretokenizer and GPT4Pretokenizer.
    """
    return (Int(BYTE_CLASS[b]) & BC_WS) != 0


# ===========================================================================
# PreTokenizer trait
# ===========================================================================
#
# The trait that all pre-tokenizers must implement.  BPETokenizer[PT] is
# parameterised on a type that satisfies this trait.
#
# Only one method is required: split(text: String) -> List[String].

# ── Byte mapping enum ─────────────────────────────────────────────────────
# Identifies which byte→ID mapping a pre-tokenizer family uses.
# SEQUENTIAL: rank = byte value (r50k_base, cl100k_base)
# SHUFFLED:   permuted mapping (o200k_base — rank 0 = '!' not 0x00)


@fieldwise_init
struct ByteMapping(ImplicitlyCopyable & Equatable):
    """Compile-time identifier for byte→ID mapping strategy.

    Each pre-tokenizer declares which mapping it uses via a comptime
    member.  This enables zero-cost branching in the encode hot path:
    SEQUENTIAL maps rank = byte (identity), SHUFFLED uses a 256-entry
    lookup table.
    """

    var _value: Int

    comptime SEQUENTIAL = ByteMapping(0)
    comptime SHUFFLED = ByteMapping(1)


# ── Comptime LUTs for o200k_base ──────────────────────────────────────────
# Derived from the first 256 entries of o200k_base.tiktoken.
# O200K_BYTE_TO_ID[byte_value] = token_rank
# O200K_ID_TO_BYTE[token_rank] = byte_value

comptime O200K_BYTE_TO_ID = SIMD[DType.int32, 256](
    188, 189, 190, 191, 192, 193, 194, 195,  # 0x00-0x07
    196, 197, 198, 199, 200, 201, 202, 203,  # 0x08-0x0F
    204, 205, 206, 207, 208, 209, 210, 211,  # 0x10-0x17
    212, 213, 214, 215, 216, 217, 218, 219,  # 0x18-0x1F
    220,  # 0x20 (space)
    0, 1, 2, 3, 4, 5, 6, 7,  # 0x21-0x28
    8, 9, 10, 11, 12, 13, 14, 15,  # 0x29-0x30
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,  # 0x31-0x38
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,  # 0x39-0x40
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,  # 0x41-0x48
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,  # 0x49-0x50
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,  # 0x51-0x58
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,  # 0x59-0x60
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,  # 0x61-0x68
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,  # 0x69-0x70
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,  # 0x71-0x78
    88,
    89,
    90,
    91,
    92,
    93,  # 0x79-0x7E
    221,  # 0x7F (DEL)
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,  # 0x80-0x87
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,  # 0x88-0x8F
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,  # 0x90-0x97
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,  # 0x98-0x9F
    254,  # 0xA0
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,  # 0xA1-0xA8
    102,
    103,
    104,
    105,  # 0xA9-0xAC
    255,  # 0xAD
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,  # 0xAE-0xB7
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,  # 0xB8-0xBF
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,  # 0xC0-0xC7
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,  # 0xC8-0xCF
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,  # 0xD0-0xD7
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,  # 0xD8-0xDF
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,  # 0xE0-0xE7
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,  # 0xE8-0xEF
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,  # 0xF0-0xF7
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,  # 0xF8-0xFF
    186,
    187,  # padding to 256
)

comptime O200K_ID_TO_BYTE = SIMD[DType.int32, 256](
    0x21,
    0x22,
    0x23,
    0x24,
    0x25,
    0x26,
    0x27,
    0x28,  # rank 0-7
    0x29,
    0x2A,
    0x2B,
    0x2C,
    0x2D,
    0x2E,
    0x2F,
    0x30,  # rank 8-15
    0x31,
    0x32,
    0x33,
    0x34,
    0x35,
    0x36,
    0x37,
    0x38,  # rank 16-23
    0x39,
    0x3A,
    0x3B,
    0x3C,
    0x3D,
    0x3E,
    0x3F,
    0x40,  # rank 24-31
    0x41,
    0x42,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,  # rank 32-39
    0x49,
    0x4A,
    0x4B,
    0x4C,
    0x4D,
    0x4E,
    0x4F,
    0x50,  # rank 40-47
    0x51,
    0x52,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,  # rank 48-55
    0x59,
    0x5A,
    0x5B,
    0x5C,
    0x5D,
    0x5E,
    0x5F,
    0x60,  # rank 56-63
    0x61,
    0x62,
    0x63,
    0x64,
    0x65,
    0x66,
    0x67,
    0x68,  # rank 64-71
    0x69,
    0x6A,
    0x6B,
    0x6C,
    0x6D,
    0x6E,
    0x6F,
    0x70,  # rank 72-79
    0x71,
    0x72,
    0x73,
    0x74,
    0x75,
    0x76,
    0x77,
    0x78,  # rank 80-87
    0x79,
    0x7A,
    0x7B,
    0x7C,
    0x7D,
    0x7E,  # rank 88-93
    0xA1,
    0xA2,
    0xA3,
    0xA4,
    0xA5,
    0xA6,
    0xA7,
    0xA8,  # rank 94-101
    0xA9,
    0xAA,
    0xAB,
    0xAC,
    0xAE,
    0xAF,
    0xB0,
    0xB1,  # rank 102-109
    0xB2,
    0xB3,
    0xB4,
    0xB5,
    0xB6,
    0xB7,
    0xB8,
    0xB9,  # rank 110-117
    0xBA,
    0xBB,
    0xBC,
    0xBD,
    0xBE,
    0xBF,
    0xC0,
    0xC1,  # rank 118-125
    0xC2,
    0xC3,
    0xC4,
    0xC5,
    0xC6,
    0xC7,
    0xC8,
    0xC9,  # rank 126-133
    0xCA,
    0xCB,
    0xCC,
    0xCD,
    0xCE,
    0xCF,
    0xD0,
    0xD1,  # rank 134-141
    0xD2,
    0xD3,
    0xD4,
    0xD5,
    0xD6,
    0xD7,
    0xD8,
    0xD9,  # rank 142-149
    0xDA,
    0xDB,
    0xDC,
    0xDD,
    0xDE,
    0xDF,
    0xE0,
    0xE1,  # rank 150-157
    0xE2,
    0xE3,
    0xE4,
    0xE5,
    0xE6,
    0xE7,
    0xE8,
    0xE9,  # rank 158-165
    0xEA,
    0xEB,
    0xEC,
    0xED,
    0xEE,
    0xEF,
    0xF0,
    0xF1,  # rank 166-173
    0xF2,
    0xF3,
    0xF4,
    0xF5,
    0xF6,
    0xF7,
    0xF8,
    0xF9,  # rank 174-181
    0xFA,
    0xFB,
    0xFC,
    0xFD,
    0xFE,
    0xFF,  # rank 182-187
    0x00,
    0x01,
    0x02,
    0x03,
    0x04,
    0x05,
    0x06,
    0x07,  # rank 188-195
    0x08,
    0x09,
    0x0A,
    0x0B,
    0x0C,
    0x0D,
    0x0E,
    0x0F,  # rank 196-203
    0x10,
    0x11,
    0x12,
    0x13,
    0x14,
    0x15,
    0x16,
    0x17,  # rank 204-211
    0x18,
    0x19,
    0x1A,
    0x1B,
    0x1C,
    0x1D,
    0x1E,
    0x1F,  # rank 212-219
    0x20,  # rank 220
    0x7F,
    0x80,
    0x81,
    0x82,
    0x83,
    0x84,
    0x85,
    0x86,  # rank 221-228
    0x87,
    0x88,
    0x89,
    0x8A,
    0x8B,
    0x8C,
    0x8D,
    0x8E,  # rank 229-236
    0x8F,
    0x90,
    0x91,
    0x92,
    0x93,
    0x94,
    0x95,
    0x96,  # rank 237-244
    0x97,
    0x98,
    0x99,
    0x9A,
    0x9B,
    0x9C,
    0x9D,
    0x9E,  # rank 245-252
    0x9F,
    0xA0,
    0xAD,  # rank 253-255
)

# ===========================================================================
# WordCounts — fused word-frequency table for training
# ===========================================================================
#
# train() used to materialize every word as a String and count frequencies
# in a Dict[String, Int] — one heap allocation per word plus string-keyed
# hashing, on top of the split chain.  WordCounts fuses counting into the
# matcher pass: the matcher hands it a (pointer, length) span, it hashes
# the bytes in place (zero copies) and increments the count in an
# open-addressing table.  Entries are assigned entry numbers 0, 1, 2, ...
# in first-seen order — the same guarantee as Dict's documented insertion
# order — so merge tie-breaking stays byte-for-byte reproducible.

from bpe.shared import IntArray, ByteArray, TokenSpan, ByteSpanArena

struct WordCounts(ImplicitlyCopyable & Movable):
    """Insertion-ordered word -> frequency table keyed by raw bytes.

    Purpose:
        Counts word frequencies during BPE training without the cost of
        Dict[String, Int] -- no heap allocation per word, no String
        construction, no string-keyed hashing. Instead, callers hand this
        struct raw (pointer, length) byte spans directly from the source
        text; it hashes those bytes in place (zero copies of the word
        itself happen until it's confirmed to be new) and tracks counts
        in a custom open-addressing hash table.

    How it works, at a glance:
        Every UNIQUE word gets one "entry" -- an integer index 0, 1, 2, ...
        assigned in first-seen order. An entry's data is spread across a
        ByteSpanArena (word bytes) plus a frequency array, all indexed by
        that same entry number:

            entry e's word bytes  = arena.bytes[arena.spans[e].offset :
                                                + arena.spans[e].length]
            entry e's frequency   = counts[e]

        `slots` is a separate open-addressing hash table (linear probing)
        that maps a word's hash to its entry number, so add() can find an
        existing word in O(1) average time instead of scanning every
        entry. Entries are numbered in first-seen order, and entry number
        IS iteration order -- no separate `order` array is needed (unlike
        Dict, whose bucket placement is arbitrary), so train() iterates
        `range(n_entries)` directly for reproducible first-seen-order
        traversal, which matters for deterministic merge tie-breaking
        during BPE training.

    Fields (all public; train() walks them directly when building the
    per-word arena):

        slots      Open-addressing hash table. slots[i] == 0 means slot i
                   is empty. slots[i] == e + 1 means slot i holds entry e
                   (stored as e+1, not e, specifically so 0 can mean
                   "empty" without colliding with the valid entry index 0).
                   Always sized as a power of two so that
                   `hash & (slot_cap - 1)` can be used instead of `%`
                   (bitmask is cheaper than modulo, and only valid because
                   slot_cap is a power of two everywhere it's used).

        arena      ByteSpanArena holding every unique word's raw bytes
                   back to back, each exactly once, with a per-entry
                   (offset, length) span. Individual words are never
                   stored more than once even though add() may be called
                   on the same word thousands of times -- only counts[e]
                   increments on repeat sightings.

        counts     entry -> how many times that word has been seen so far
                   (this IS the frequency table -- everything else exists
                   to compute and index into this array correctly).

        n_entries  Total number of unique words seen so far. Also the
                   next entry number that will be assigned.

        slot_cap   Current size of the `slots` array (always a power of
                   two). Grows via _rehash() as n_entries approaches it.

    Typical usage (from train()):
        var wc = WordCounts()
        for each word span found while splitting the corpus:
            wc.add(word_ptr, word_len)
        # afterwards, for e in range(wc.n_entries): wc.counts[e] is that
        # word's frequency, and wc.arena.bytes[wc.arena.spans[e].offset :
        # +wc.arena.spans[e].length] is its raw bytes.
    """

    comptime FNV_offset_basis = UInt64(0xCBF29CE484222325)
    """Starting hash value for FNV-1a (a fixed constant defined by the
    FNV-1a algorithm itself -- not arbitrary, must match the spec exactly
    for the hash to behave correctly)."""

    comptime FNV_prime = UInt64(0x100000001B3)
    """Multiplier constant for FNV-1a (also fixed by the algorithm's
    definition -- chosen for good bit-mixing properties, not a value
    that can be changed without changing the hash function's behavior)."""

    var arena: ByteSpanArena
    var slots: IntArray
    var counts: IntArray
    var n_entries: Int
    var slot_cap: Int

    def __init__(out self, capacity: Int = 4096):
        """Create an empty table sized for roughly `capacity` unique words.

        `capacity` is a hint, not a hard limit -- the table grows
        automatically via _rehash() if more unique words arrive than
        expected; this just avoids early reallocation for the common case.

        The hash table itself (`slots`) is sized to 4x `capacity`, rounded
        up to the next power of two, so that it starts at a 25% load
        factor (25% of slots occupied once `capacity` entries are added).
        Starting this sparse keeps collision rates low and probe chains
        short even before any rehash has occurred.

        Example: capacity=4096 -> cap starts at 16, doubles until it's
        >= 16384 (4096*4) -> slot_cap ends up 16384.

        Args:
            capacity: Expected number of unique words. Default 4096.
        """
        var cap = 16
        while cap < capacity * 4:
            cap *= 2
        self.slot_cap = cap
        self.slots = IntArray(length=cap, fill=0)  # 0 = every slot empty
        self.arena = ByteSpanArena()
        self.arena.spans.reserve(capacity)
        self.counts = IntArray(capacity=capacity)
        self.n_entries = 0

    def __init__(out self, *, copy: Self):
        """Deep-copy constructor -- every backing array is independently
        duplicated, so mutating the copy never affects the original and
        vice versa."""
        self.slot_cap = copy.slot_cap
        self.slots = copy.slots.copy()
        self.arena = ByteSpanArena(copy=copy.arena)
        self.counts = copy.counts.copy()
        self.n_entries = copy.n_entries

    def __init__(out self, *, deinit move: Self):
        """Move constructor -- takes ownership of `move`'s backing arrays
        directly (the `^` transfer sigil) rather than copying their
        contents, so this is O(1) regardless of table size. `move` is
        consumed and can't be used afterward."""
        self.slot_cap = move.slot_cap
        self.slots = move.slots^
        self.arena = move.arena^
        self.counts = move.counts^
        self.n_entries = move.n_entries

    def _rehash(mut self, new_cap: Int):
        """Grow the hash table to `new_cap` slots and reinsert every
        existing entry into it.

        Only the `slots` array is rebuilt here -- arena/counts are
        untouched, since entry numbers and their data never change across
        a rehash, only *where in `slots` each entry's pointer lives*
        changes (because the bucket index depends on
        `hash & (slot_cap - 1)`, which shifts once slot_cap changes).

        Called automatically by add() once the table gets too full (see
        add()'s load-factor check) -- not intended to be called directly.

        Args:
            new_cap: The new slot array size. Must be a power of two,
                since bucket indices are computed with a bitmask
                (`& (new_cap - 1)`) rather than modulo.
        """
        self.slots = IntArray(length=new_cap, fill=0)
        self.slot_cap = new_cap
        var nsp = self.slots.unsafe_ptr()
        var bp = self.arena.bytes.unsafe_ptr()
        # Re-walk every existing entry (in entry-number order) and
        # reinsert it into the freshly-sized, freshly-zeroed slots array.
        for e in range(self.n_entries):
            var h = Self._fnv1a64(
                bp + self.arena.spans[e].offset, self.arena.spans[e].length
            )
            var idx = Int(h & UInt64(new_cap - 1))
            # Linear probe forward until an empty slot is found -- same
            # probing rule used in add(), so lookups after a rehash land
            # in exactly the same relative positions they would if the
            # table had always been this size.
            while nsp[idx] != 0:
                idx = (idx + 1) & (new_cap - 1)
            nsp[idx] = e + 1  # store as e+1; see `slots` field doc for why

    def add[
        origin: Origin, //
    ](mut self, ptr: UnsafePointer[UInt8, origin], length: Int):
        """Record one occurrence of the word at ptr[0:length].

        If this exact byte sequence has been seen before, its existing
        entry's count is incremented by 1 (O(1) average case, no
        allocation). If it's new, a new entry is created: its bytes are
        copied once into the arena, and counts/slots are updated to track
        it.

        Zero-length words are silently ignored (a no-op) rather than
        treated as a valid empty-string entry.

        Growth: before inserting, checks whether the table has crossed a
        50% load factor (n_entries*2 >= slot_cap) and doubles the slot
        array via _rehash() if so -- this keeps average probe-chain
        length short as the table fills, at the cost of an O(n) rehash
        pass on the (amortized rare) occasions it triggers.

        Lookup: hashes the incoming bytes with FNV-1a, then walks the
        slots array starting at `hash & (slot_cap - 1)`, following the
        linear-probe chain (checking the next slot whenever the current
        one is occupied) until either:
          (a) an occupied slot's entry has matching length AND matching
              bytes (verified with memcmp, since hash equality alone
              doesn't guarantee the bytes are actually the same word --
              this is genuine collision handling, not just an
              optimization), in which case that entry's count is bumped
              and the function returns, or
          (b) an empty slot (value 0) is reached, meaning this word has
              never been seen -- a new entry is created here.

        Args:
            ptr: Pointer to the first byte of the word.
            length: Number of bytes in the word. 0 is a no-op.
        """
        if length == 0:
            return
        if self.n_entries * 2 >= self.slot_cap:
            self._rehash(self.slot_cap * 2)
        var h = Self._fnv1a64(ptr, length)
        # Hoist every field chain the probe loop reads.  arena.spans,
        # arena.bytes and counts are only reallocated by the new-entry
        # path AFTER the probe loop exits (arena.add / counts.append
        # below), so these raw pointers stay valid throughout the loop
        # -- and because LLVM sees realloc-capable calls in the loop
        # body it would otherwise reload the chains on every probe step.
        var slot_cap = self.slot_cap
        var sp = self.slots.unsafe_ptr()
        var spans_ptr = self.arena.spans.unsafe_ptr()
        var bytes_ptr = self.arena.bytes.unsafe_ptr().as_noalias_ptr()
        var counts_ptr = self.counts.unsafe_ptr()
        var idx = Int(h & UInt64(slot_cap - 1))
        while sp[idx] != 0:
            var e = sp[idx] - 1  # stored entries are offset by +1; undo it
            if spans_ptr[e].length == length and memcmp(
                bytes_ptr + spans_ptr[e].offset,
                ptr,
                length,
            ) == 0:
                # Found: same length and byte-identical -> same word.
                counts_ptr[e] = counts_ptr[e] + 1
                return
            # Occupied by a different word (hash collision) -> probe next.
            idx = (idx + 1) & (slot_cap - 1)
        # Reached an empty slot without finding a match -> genuinely new
        # word. Create entry `e`, copy its bytes into the arena once,
        # and register it in the frequency array plus the hash table.
        var e = self.n_entries
        _ = self.arena.add(ptr, length)
        sp[idx] = e + 1
        self.counts.append(1)
        self.n_entries += 1

    def add[origin: Origin, //](mut self, span: Span[UInt8, origin]):
        """Convenience overload: record one occurrence of the word held
        in `span`. Equivalent to add(span.unsafe_ptr(), len(span))."""
        self.add(span.unsafe_ptr(), len(span))

    def add[
        origin: Origin, //
    ](mut self, start: UnsafePointer[UInt8, origin], pos: Int, length: Int):
        """Convenience overload: record one occurrence of the word found
        at byte offset `pos` within the buffer starting at `start`.
        Equivalent to add(start + pos, length) -- useful when the caller
        is walking offsets into one larger buffer rather than holding a
        separate pointer per word."""
        self.add(start + pos, length)

    @staticmethod
    @always_inline
    def _fnv1a64[
        origin: Origin, //
    ](ptr: UnsafePointer[UInt8, origin], n: Int) -> UInt64:
        """Compute the 64-bit FNV-1a hash of n bytes starting at ptr.

        FNV-1a processes one byte at a time: XOR the byte into the
        running hash, then multiply by the FNV prime. This ordering
        (XOR-then-multiply, as opposed to FNV-1's multiply-then-XOR) is
        what gives FNV-1a its slightly better bit-avalanche behavior for
        short keys like individual words -- appropriate here since most
        words are just a handful of bytes.

        Not cryptographically secure and not intended to be -- this is a
        fast, well-distributed hash for hash-table bucketing, not for any
        security-sensitive use.

        Args:
            ptr: Pointer to the first byte to hash.
            n: Number of bytes to hash.

        Returns:
            The 64-bit FNV-1a hash of ptr[0:n].
        """
        var h: UInt64 = Self.FNV_offset_basis
        for i in range(n):
            h ^= UInt64(ptr[i])
            h *= Self.FNV_prime
        return h


trait PreTokenizer(Movable & Defaultable & ImplicitlyDeletable & Writable):
    """Split raw text into "words" for BPE training and encoding.

    Any struct implementing this trait can be used as the PT parameter
    of BPETokenizer[PT: PreTokenizer].  The split methods divide the
    input text into the atomic units that the BPE merge algorithm
    operates on.

    The trait provides two entry points with different performance
    characteristics:

        split_view
          Returns zero-copy StringSlice views into the original input.
          No heap allocation per word.  Used by encode_ordinary() and
          encode() — the hot path.

        split
          Returns owned String copies.  Default implementation wraps
          split_view into String allocations.  Used by train() where
          words become Dict keys (legacy — train() now uses
          count_words, which never materializes words).

    Pre-tokenizers that only slice the input (GPT-2 r50k_base, GPT-4
    cl100k_base / o200k_base) override split_view for maximum encode
    throughput.

    Byte mapping
    ------------
    Each pre-tokenizer declares which byte→ID mapping it uses via the
    comptime ``byte_map`` member.  The ``byte_to_id`` and ``id_to_byte``
    static methods convert between raw byte values (0-255) and base
    token ranks.  For SEQUENTIAL mapping (r50k, cl100k), rank = byte.
    For SHUFFLED mapping (o200k), a 256-entry lookup table is used.

    Shared whitespace matchers
    --------------------------
    The trait provides three default static methods that implement the
    whitespace-matching logic common to both GPT-2 r50k_base and GPT-4
    cl100k_base pre-tokenizers.  Family-specific matchers (contractions,
    letter runs, digit runs, punctuation runs, newlines) remain on
    each concrete implementation.
    """

    # ── byte mapping (required comptime member) ──────────────────────
    comptime byte_map: ByteMapping

    @staticmethod
    @always_inline
    def byte_to_id(b: Int) -> Int:
        """Convert a raw byte value (0-255) to its base token rank.

        Default: identity mapping (rank = byte).  Override for shuffled
        mappings (e.g., o200k_base).
        """
        return b

    @staticmethod
    @always_inline
    def id_to_byte(rank: Int) -> Int:
        """Convert a base token rank (0-255) back to its raw byte value.

        Default: identity mapping (byte = rank).  Override for shuffled
        mappings (e.g., o200k_base).
        """
        return rank

    def split_view[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin]) raises -> List[StringSlice[origin]]:
        """Split text into zero-copy word views.

        Returns views into the original ``text`` — no per-word heap
        allocation.  The views are valid only for the lifetime of the
        input ``text``.  Used by the encode hot path.

        Default: return the entire input as a single word (correct for
        pre-tokenizers that don't subdivide text).

        Override in pre-tokenizers that slice the input (GPT2Pretokenizer,
        GPT4Pretokenizer) for zero-allocation encode.
        """
        var result = List[StringSlice[origin]]()
        result.reserve(1)
        result.append(text)
        return result^

    def split[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin]) raises -> List[String]:
        """Split text into owned word strings.

        Default implementation wraps split_view into String allocations.

        Legacy API — train() uses count_words() (below), which never
        materializes words.
        """
        var views = self.split_view(text)
        var result = List[String](capacity=len(views))
        for v in views:
            result.append(String(v))
        return result^

    def count_words[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin], mut counts: WordCounts) raises:
        """Count word frequencies in one fused pass over ``text``.

        The training hot path: instead of split() + Dict[String, Int]
        counting (one String allocation per word), the matcher hands each
        word span straight to ``counts``, which hashes the bytes in place.
        The default implementation delegates to split_view; GPT2Pretokenizer
        and GPT4Pretokenizer override with an inlined matcher loop.
        """
        ref views = self.split_view(text)
        for v in views:
            counts.add(v.as_bytes())

    @staticmethod
    def name() -> String:
        ...

    @staticmethod
    def special_tokens() -> Dict[String, Int]:
        ...

    # ── shared whitespace matchers (default implementations) ─────────
    # These are functionally identical between GPT-2 r50k_base and GPT-4
    # cl100k_base pre-tokenizers.  Family-specific matchers (contractions,
    # letter/digit/punct runs, newlines) remain on each concrete struct.

    @staticmethod
    def match_trailing_all_ws[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match alternative 5: \\s++$.

        Matches if ALL remaining bytes from pos to end-of-string are
        ASCII whitespace.  Returns the match length (entire remaining
        span) or 0 if any non-whitespace byte follows.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var i = pos
        var count = 0
        while i < n:
            if not is_ascii_ws_byte(Int(span[i])):
                return 0
            count += 1
            i += 1
        return count

    @staticmethod
    def match_ws_not_before_nonws[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match whitespace run where the byte AFTER is also whitespace (or EOS).

        Approximates the regex negative lookahead (?!\\S).  Scans the
        full ASCII-whitespace run, then shrinks from the right until
        the post-match byte IS whitespace or end-of-string.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        if pos >= n:
            return 0
        if not is_ascii_ws_byte(Int(span[pos])):
            return 0
        var end = pos
        while end < n and is_ascii_ws_byte(Int(span[end])):
            end += 1
        var total = end - pos
        var ws_len = total
        while ws_len >= 1:
            var next_pos = pos + ws_len
            if next_pos >= n or is_ascii_ws_byte(Int(span[next_pos])):
                return ws_len
            ws_len -= 1
        return 0

    @staticmethod
    def match_single_ws[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match a single ASCII whitespace byte.

        Catch-all fallback for whitespace bytes not matched by the
        trailing-ws or ws-before-nonws alternatives.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            1 if whitespace, 0 otherwise.
        """
        if pos >= len(span):
            return 0
        if is_ascii_ws_byte(Int(span[pos])):
            return 1
        return 0


# ===========================================================================
# GPT-2 / r50k_base Pre-tokenizer
# ===========================================================================
#
# Matches tiktoken's GPT-2 byte-level pre-tokenizer exactly.
#
# Pattern (7 alternatives, tried left-to-right):
#
#   '(?:[sdmt]|ll|ve|re)          # 1  Apostrophe contractions
#   | ?\p{L}++                     # 2  Optional space + letter run
#   | ?\p{N}++                     # 3  Optional space + digit run
#   | ?[^\s\p{L}\p{N}]++           # 4  Optional space + punctuation run
#   | \s++$                        # 5  Trailing whitespace at EOS
#   | \s+(?!\S)                    # 6  Ws not followed by non-space
#   | \s                           # 7  Single whitespace (fallback)
#
# All whitespace is kept as-is in the output tokens -- no G substitution.


struct GPT2Pretokenizer(PreTokenizer):
    comptime byte_map: ByteMapping = ByteMapping.SEQUENTIAL

    def __init__(out self):
        pass

    @staticmethod
    def name() -> String:
        return String("gpt2")

    @staticmethod
    def special_tokens() -> Dict[String, Int]:
        var d = Dict[String, Int]()
        d["<|endoftext|>"] = 50256
        return d^

    @staticmethod
    @always_inline
    def _match_contraction[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match alternative 1: '(?:[sdmt]|ll|ve|re).

        Looks for a leading apostrophe (byte 0x27), then checks for
        known contraction suffixes.  Case-sensitive -- only ASCII
        lowercase letters are accepted.

        Matches: 's, 't, 'm, 'd, 'll, 've, 're.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        if pos >= n or span[pos] != UInt8(39):
            return 0
        if pos + 2 > n:
            return 0
        var c = Int(span[pos + 1])
        if c == 115 or c == 116 or c == 109 or c == 100:
            return 2
        if pos + 3 <= n:
            var c1 = Int(span[pos + 2])
            if c == 108 and c1 == 108:
                return 3
            if c == 118 and c1 == 101:
                return 3
            if c == 114 and c1 == 101:
                return 3
        return 0

    @staticmethod
    @always_inline
    def _match_letter_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match alternative 2:  ?\\p{L}++.

        Consumes an optional ASCII space (0x20), then greedily consumes
        consecutive Unicode letters.  Returns 0 if no letters found
        after the optional space.

        Corresponds to regex: an optional space followed by one or more
        Unicode letters (possessive, no backtracking).

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        if span[i] == UInt8(32):
            i += 1
        var found = False
        var n_run = _swar_letter_run(span, i, n)
        i += n_run
        if n_run > 0:
            found = True
        while i < n:
            var b = span[i]
            if b < 0x80:
                if Int(BYTE_CLASS[Int(b)]) & BC_LETTER:
                    found = True
                    i += 1
                else:
                    break
            else:
                var cur_len = utf8_byte_length(b)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_letter(cur_cp):
                    found = True
                    i += cur_len
                else:
                    break
        if not found:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_digit_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match alternative 3:  ?\\p{N}++.

        Consumes an optional ASCII space (0x20), then greedily consumes
        consecutive Unicode digits.  No maximum digit count (unlike
        GPT-4 which caps at 3).

        Corresponds to regex: an optional space followed by one or more
        Unicode digits (possessive).

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        if span[i] == UInt8(32):
            i += 1
        var found = False
        while i < n:
            var b = span[i]
            if b < 0x80:
                if Int(BYTE_CLASS[Int(b)]) & BC_DIGIT:
                    found = True
                    i += 1
                else:
                    break
            else:
                var cur_len = utf8_byte_length(b)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_digit(cur_cp):
                    found = True
                    i += cur_len
                else:
                    break
        if not found:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_punct_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match alternative 4:  ?[^\\s\\p{L}\\p{N}]++.

        Consumes an optional ASCII space (0x20), then greedily consumes
        consecutive characters that are NOT whitespace, letters, or
        digits -- i.e., punctuation, symbols, operators, emoji, etc.

        No trailing newline consumption (unlike GPT-4 which adds
        possessive trailing newlines after the punct run).

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        if span[i] == UInt8(32):
            i += 1
        var found = False
        while i < n:
            var b = span[i]
            if b < 0x80:
                if Int(BYTE_CLASS[Int(b)]) & (BC_WS | BC_LETTER | BC_DIGIT):
                    break
                found = True
                i += 1
            else:
                var cur_len = utf8_byte_length(b)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_whitespace(cur_cp) or is_letter_or_digit(cur_cp):
                    break
                found = True
                i += cur_len
        if not found:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _best_match[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Try all 7 matchers left-to-right; return the first match.

        Implements regex alternation: matchers are tried in order
        (alternatives 1-7), and the FIRST one that returns a positive
        match length wins.  This is standard regex alternation
        semantics, NOT "longest match wins".

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            The match length in bytes, or 0 if no matcher matched.
            When 0 is returned, the caller (split()) falls back to
            consuming a single codepoint via utf8_byte_length().
        """
        var m = GPT2Pretokenizer._match_contraction(span, pos)
        if m > 0:
            return m
        m = GPT2Pretokenizer._match_letter_run(span, pos)
        if m > 0:
            return m
        m = GPT2Pretokenizer._match_digit_run(span, pos)
        if m > 0:
            return m
        m = GPT2Pretokenizer._match_punct_run(span, pos)
        if m > 0:
            return m
        m = Self.match_trailing_all_ws(span, pos)
        if m > 0:
            return m
        m = Self.match_ws_not_before_nonws(span, pos)
        if m > 0:
            return m
        return Self.match_single_ws(span, pos)

    def split_view[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin]) raises -> List[StringSlice[origin]]:
        """Split text into zero-copy word views (GPT-2 r50k_base).

        No per-word heap allocation — each entry is a view into the
        original ``text`` input.
        """
        var result = List[StringSlice[origin]]()
        var n = text.byte_length()
        if n == 0:
            return result^
        var span = text.as_bytes()
        var pos = 0
        while pos < n:
            var best_len = GPT2Pretokenizer._best_match(span, pos)
            if best_len == 0:
                best_len = utf8_byte_length(span[pos])
            var byte_span = span[pos : pos + best_len]
            result.append(StringSlice(unsafe_from_utf8=byte_span))
            pos += best_len
        return result^

    def count_words[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin], mut counts: WordCounts) raises:
        """Fused word counting: same matcher loop as split_view, but each
        matched span goes straight into ``counts`` — no StringSlice list,
        no String materialization.
        """
        var n = text.byte_length()
        if n == 0:
            return
        var span = text.as_bytes()
        var sp = span.unsafe_ptr()
        var pos = 0
        while pos < n:
            var best_len = GPT2Pretokenizer._best_match(span, pos)
            if best_len == 0:
                best_len = utf8_byte_length(sp[pos])
            counts.add(sp + pos, best_len)
            pos += best_len

    def write_to[T: Writer](self, mut writer: T):
        writer.write(String("GPT2Pretokenizer"))


# ===========================================================================
# GPT-4 / cl100k_base Pre-tokenizer
# ===========================================================================
#
# Matches tiktoken's GPT-4 byte-level pre-tokenizer exactly.
#
# Pattern (8 alternatives, tried left-to-right):
#
#   '(?i:[sdmt]|ll|ve|re)           # 1  Case-insensitive contractions
#   | [^\r\n\p{L}\p{N}]?+\p{L}++    # 2  Optional prefix + letter run
#   | \p{N}{1,3}+                    # 3  Digit run, max 3 codepoints
#   |  ?[^\s\p{L}\p{N}]++[\r\n]*+   # 4  Space + punct + trailing newlines
#   | \s++$                          # 5  Trailing whitespace at EOS
#   | \s*[\r\n]                      # 6  Optional ws + one newline char
#   | \s+(?!\S)                      # 7  Ws not followed by non-space
#   | \s                             # 8  Single whitespace fallback
#
# Key differences from GPT-2:
#   - Contractions are case-insensitive (matches 'S, 'T, 'LL, etc.)
#   - Letter run has optional non-letter/digit/nl prefix char
#   - Digit run capped at 3 digits (no space prefix)
#   - Punct run can consume trailing CR and LF
#   - Explicit newline matcher: \s*[\r\n]
#   - 8 alternatives (vs 7 for GPT-2)


struct GPT4Pretokenizer[
    mapping: ByteMapping = ByteMapping.SEQUENTIAL,
](PreTokenizer):
    comptime byte_map: ByteMapping = Self.mapping

    def __init__(out self):
        pass

    @staticmethod
    def name() -> String:
        comptime if Self.mapping == ByteMapping.SEQUENTIAL:
            return String("cl100k")
        else:
            return String("o200k")

    @staticmethod
    def special_tokens() -> Dict[String, Int]:
        comptime if Self.mapping == ByteMapping.SEQUENTIAL:
            var d = Dict[String, Int]()
            d["<|endoftext|>"] = 100257
            d["<|fim_prefix|>"] = 100258
            d["<|fim_middle|>"] = 100259
            d["<|fim_suffix|>"] = 100260
            d["<|endofprompt|>"] = 100276
            return d^
        else:
            var d = Dict[String, Int]()
            d["<|endoftext|>"] = 199999
            d["<|endofprompt|>"] = 200018
            return d^

    @staticmethod
    def byte_to_id(b: Int) -> Int:
        """Raw byte (0-255) → base token rank.

        For SEQUENTIAL: identity (rank = byte).
        For SHUFFLED (o200k): 256-entry comptime LUT.
        """
        comptime if Self.mapping == ByteMapping.SHUFFLED:
            return Int(O200K_BYTE_TO_ID[b])
        else:
            return b

    @staticmethod
    def id_to_byte(rank: Int) -> Int:
        """Base token rank → raw byte (0-255).

        For SEQUENTIAL: identity (byte = rank).
        For SHUFFLED (o200k): 256-entry comptime LUT.
        """
        comptime if Self.mapping == ByteMapping.SHUFFLED:
            return Int(O200K_ID_TO_BYTE[rank])
        else:
            return rank

    @staticmethod
    @always_inline
    def _match_contraction[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match alternative 1: '(?i:[sdmt]|ll|ve|re).

        Same as GPT-2 but case-insensitive: letters in the contraction
        suffix are lowered via OR with 0x20 before comparison.

        Matches: 's, 'S, 't, 'T, 'm, 'M, 'd, 'D, 'll, 'LL, 'lL, 'Ll,
        've, 'VE, 'vE, 'Ve, 're, 'RE, 'rE, 'Re.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        if pos >= n or span[pos] != UInt8(39):
            return 0
        if pos + 2 > n:
            return 0
        var c = Int(span[pos + 1]) | 0x20
        if c == 115 or c == 116 or c == 109 or c == 100:
            return 2
        if pos + 3 <= n:
            var c1 = Int(span[pos + 2]) | 0x20
            if c == 108 and c1 == 108:
                return 3
            if c == 118 and c1 == 101:
                return 3
            if c == 114 and c1 == 101:
                return 3
        return 0

    @staticmethod
    @always_inline
    def _match_letter_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match alternative 2: [^\\r\\n\\p{L}\\p{N}]?+\\p{L}++.

        Unlike GPT-2's "optional space + letters", this matcher allows
        an optional SINGLE prefix character that is NOT:
          - CR (carriage return)
          - LF (newline)
          - a letter
          - a digit

        After the optional prefix, greedily consumes consecutive
        letters.  Must find at least one letter (otherwise returns 0).

        Examples:
          "$hello"  -> matched as one token (prefix '$' + letters)
          "42abc"  -> NOT matched here (digits match alt 3 first)
          "\\nhello" -> NOT matched (CR/LF prefix excluded)

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        var lead = span[i]
        if lead < 0x80:
            if (
                Int(BYTE_CLASS[Int(lead)]) & (BC_LETTER | BC_DIGIT | BC_CRLF)
            ) == 0:
                i += 1
        else:
            var cplen = utf8_byte_length(lead)
            var cp = decode_codepoint(span.unsafe_ptr() + i, cplen)
            if cp != 0x000A and cp != 0x000D and not is_letter_or_digit(cp):
                i += cplen
        var found_letters = False
        var n_run = _swar_letter_run(span, i, n)
        i += n_run
        if n_run > 0:
            found_letters = True
        while i < n:
            var b = span[i]
            if b < 0x80:
                if Int(BYTE_CLASS[Int(b)]) & BC_LETTER:
                    found_letters = True
                    i += 1
                else:
                    break
            else:
                var cur_len = utf8_byte_length(b)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_letter(cur_cp):
                    found_letters = True
                    i += cur_len
                else:
                    break
        if not found_letters:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_digit_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match alternative 3: \\p{N}{1,3}+.

        Matches 1 to 3 consecutive Unicode digits.  No space prefix
        (unlike GPT-2).  Possessive -- once 3 digits are found, stops
        even if more digits follow.

        Examples:
          "123"   -> matched as "123"
          "1234"  -> matched as "123" (4th digit left for next iter)
          "42"    -> matched as "42"
          "abc"   -> 0 (no digits)

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var i = pos
        var count = 0
        while i < n and count < 3:
            var b = span[i]
            if b < 0x80:
                if Int(BYTE_CLASS[Int(b)]) & BC_DIGIT:
                    count += 1
                    i += 1
                else:
                    break
            else:
                var cplen = utf8_byte_length(b)
                var cp = decode_codepoint(span.unsafe_ptr() + i, cplen)
                if is_digit(cp):
                    count += 1
                    i += cplen
                else:
                    break
        if count == 0:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_punct_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match alternative 4:  ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*+.

        Like GPT-2 alt 4 but with a key addition: trailing CR and LF
        characters are consumed possessively after the punctuation run.

        Algorithm:
          1. Optionally consume an ASCII space (0x20)
          2. Greedily consume non-ws, non-letter, non-digit chars
          3. Greedily consume trailing CR and LF bytes

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        if span[i] == UInt8(32):
            i += 1
        var found_punct = False
        while i < n:
            var b = span[i]
            if b < 0x80:
                if Int(BYTE_CLASS[Int(b)]) & (BC_WS | BC_LETTER | BC_DIGIT):
                    break
                found_punct = True
                i += 1
            else:
                var cplen = utf8_byte_length(b)
                var cp = decode_codepoint(span.unsafe_ptr() + i, cplen)
                if is_whitespace(cp) or is_letter_or_digit(cp):
                    break
                found_punct = True
                i += cplen
        if not found_punct:
            return 0
        while i < n and (span[i] == UInt8(10) or span[i] == UInt8(13)):
            i += 1
        return i - pos

    @staticmethod
    @always_inline
    def _match_newline[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match alternative 6: \\s*[\\r\\n].

        Matches optional leading ASCII whitespace then exactly one CR
        or LF as the last character.  Implements \\s*[\\r\\n] where:
          - \\s* is greedy (consumes all ASCII whitespace)
          - [\\r\\n] must be the final character

        Algorithm:
          1. Scan the full ASCII-ws run starting at pos
          2. Shrink from the right until the last byte is CR or LF
          3. If found, return that length; if no CR/LF after shrinking
             below 1, return 0

        Examples:
          "\\n"     -> "\\n"
          " \\n"    -> " \\n"
          "  \\r\\n" -> "  \\r"  (only one CR/LF captured)

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        if pos >= n:
            return 0
        var end = pos
        while end < n and is_ascii_ws_byte(Int(span[end])):
            end += 1
        var total = end - pos
        if total == 0:
            return 0
        var ws_len = total
        while ws_len >= 1:
            var last_b = Int(span[pos + ws_len - 1])
            if last_b == 10 or last_b == 13:
                return ws_len
            ws_len -= 1
        return 0

    # ── o200k_base matchers ──────────────────────────────────────────────
    # tiktoken's o200k_base uses a different pre-tokenizer pattern from
    # cl100k_base.  The 7 alternatives are:
    #
    #   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*  # 1  Prefix + upper* + lower+ + (?i:contraction)
    #     [\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    #   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+  # 2  Prefix + upper+ + lower* + (?i:contraction)
    #     [\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    #   | \p{N}{1,3}                                           # 3  Digit run, max 3
    #   |  ?[^\s\p{L}\p{N}]+[\r\n/]*                          # 4  Space + punct + trailing nl/slash
    #   | \s*[\r\n]+                                           # 5  Ws* + one or more CR/LF
    #   | \s+(?!\S)                                            # 6  Ws not followed by non-space
    #   | \s+                                                  # 7  Whitespace run
    #
    # Key differences from cl100k:
    #   - Contractions are suffixes on letter alternatives (not standalone)
    #   - Letter runs split by case: upper-first-then-lower vs all-upper
    #   - Punct run allows trailing '/' in addition to CR/LF
    #   - Newline matcher consumes 1+ CR/LF (old consumed exactly 1)
    #   - 7 alternatives (vs 8 for cl100k — no standalone contraction)

    @staticmethod
    @always_inline
    def _match_o200k_alt1[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match o200k alternative 1: letter run with at least one lowercase.

        [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*
          [\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?

        Tries optional non-L/N prefix, then zero+ upper-like letters,
        then one+ lower-like letters (which must include at least one Ll),
        then optional contraction.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        # Optional prefix: [^\r\n\p{L}\p{N}]?
        var lead = span[i]
        if lead < 0x80:
            if (
                Int(BYTE_CLASS[Int(lead)]) & (BC_LETTER | BC_DIGIT | BC_CRLF)
            ) == 0:
                i += 1
                if i >= n:
                    return 0
        else:
            var cplen = utf8_byte_length(lead)
            var cp = decode_codepoint(span.unsafe_ptr() + i, cplen)
            if cp != 0x000A and cp != 0x000D and not is_letter_or_digit(cp):
                i += cplen
                if i >= n:
                    return 0
        # Upper-like letters: [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*
        while i < n:
            var b = span[i]
            if b < 0x80:
                if 65 <= Int(b) <= 90:
                    i += 1
                else:
                    break
            else:
                var cur_lead = b
                var cur_len = utf8_byte_length(cur_lead)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_upper_like(cur_cp):
                    i += cur_len
                else:
                    break
        # Lower-like letters: [\p{Ll}\p{Lm}\p{Lo}\p{M}]+
        var lower_start = i
        while i < n:
            var b = span[i]
            if b < 0x80:
                var ib = Int(b)
                if (65 <= ib <= 90) or (97 <= ib <= 122):
                    i += 1
                else:
                    break
            else:
                var cur_lead = b
                var cur_len = utf8_byte_length(cur_lead)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_lower_like(cur_cp):
                    i += cur_len
                else:
                    break
        if i == lower_start:
            return 0
        # Optional contraction: (?i:'s|'t|'re|'ve|'m|'ll|'d)?
        var cont = GPT4Pretokenizer._match_contraction(span, i)
        if cont > 0:
            i += cont
        return i - pos

    @staticmethod
    @always_inline
    def _match_o200k_alt2[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match o200k alternative 2: letter run with at least one uppercase.

        [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+
          [\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?

        Like alt 1 but requires 1+ upper-like letters first, then optional
        lower-like letters.  Catches all-caps words and sequences without
        a lowercase component.

        Note: alt 1 is tried first.  This alternative only fires when
        there are upper-like letters with no following lower-like letters.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        # Optional prefix: [^\r\n\p{L}\p{N}]?
        var lead = span[i]
        if lead < 0x80:
            if (
                Int(BYTE_CLASS[Int(lead)]) & (BC_LETTER | BC_DIGIT | BC_CRLF)
            ) == 0:
                i += 1
                if i >= n:
                    return 0
        else:
            var cplen = utf8_byte_length(lead)
            var cp = decode_codepoint(span.unsafe_ptr() + i, cplen)
            if cp != 0x000A and cp != 0x000D and not is_letter_or_digit(cp):
                i += cplen
                if i >= n:
                    return 0
        # Upper-like letters: [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+
        var upper_start = i
        while i < n:
            var b = span[i]
            if b < 0x80:
                if 65 <= Int(b) <= 90:
                    i += 1
                else:
                    break
            else:
                var cur_lead = b
                var cur_len = utf8_byte_length(cur_lead)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_upper_like(cur_cp):
                    i += cur_len
                else:
                    break
        if i == upper_start:
            return 0
        # Lower-like letters: [\p{Ll}\p{Lm}\p{Lo}\p{M}]*
        while i < n:
            var b = span[i]
            if b < 0x80:
                var ib = Int(b)
                if (65 <= ib <= 90) or (97 <= ib <= 122):
                    i += 1
                else:
                    break
            else:
                var cur_lead = b
                var cur_len = utf8_byte_length(cur_lead)
                var cur_cp = decode_codepoint(span.unsafe_ptr() + i, cur_len)
                if is_lower_like(cur_cp):
                    i += cur_len
                else:
                    break
        # Optional contraction: (?i:'s|'t|'re|'ve|'m|'ll|'d)?
        var cont = GPT4Pretokenizer._match_contraction(span, i)
        if cont > 0:
            i += cont
        return i - pos

    @staticmethod
    @always_inline
    def _match_o200k_punct_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Match o200k alternative 4:  ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*.

        Like cl100k alt 4 but trailing CR/LF/slash are consumed via
        [\\r\\n/]* instead of [\\r\\n]*+.
        """
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        if span[i] == UInt8(32):
            i += 1
        var found_punct = False
        while i < n:
            var b = span[i]
            if b < 0x80:
                if Int(BYTE_CLASS[Int(b)]) & (BC_WS | BC_LETTER | BC_DIGIT):
                    break
                found_punct = True
                i += 1
            else:
                var lead = b
                var cplen = utf8_byte_length(lead)
                var cp = decode_codepoint(span.unsafe_ptr() + i, cplen)
                if is_whitespace(cp) or is_letter_or_digit(cp):
                    break
                found_punct = True
                i += cplen
        if not found_punct:
            return 0
        # Trailing CR, LF, or slash
        while i < n and (
            span[i] == UInt8(10) or span[i] == UInt8(13) or span[i] == UInt8(47)
        ):
            i += 1
        return i - pos

    @staticmethod
    @always_inline
    def _match_o200k_ws_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match o200k alternative 7: \\s+ — one or more ASCII whitespace bytes.

        Unlike cl100k's final \\s (single byte), o200k uses \\s+ to consume
        an entire whitespace run in one token.
        """
        var n = len(span)
        if pos >= n:
            return 0
        if not is_ascii_ws_byte(Int(span[pos])):
            return 0
        var i = pos + 1
        while i < n and is_ascii_ws_byte(Int(span[i])):
            i += 1
        return i - pos

    @staticmethod
    @always_inline
    def _match_o200k_newline[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:
        """Match o200k alternative 5: \\s*[\\r\\n]+.

        Like cl100k alt 6 but requires 1+ CR/LF characters, not exactly 1.

        Implements regex backtracking: \\s* greedily consumes all ASCII
        whitespace, then shrinks from the right until the remaining suffix
        ends with at least one CR or LF.
        """
        var n = len(span)
        if pos >= n:
            return 0
        # Scan forward to find end of full ASCII-whitespace run
        var ws_end = pos
        while ws_end < n and is_ascii_ws_byte(Int(span[ws_end])):
            ws_end += 1
        if ws_end == pos:
            return 0
        # Scan backward from ws_end to find a CR/LF, then include all
        # consecutive CR/LF bytes after it.  This simulates the regex
        # backtracking: \\s* [\\r\\n]+  where \\s* gives up characters
        # until [\\r\\n]+ can match at least one.
        var i = ws_end
        while i > pos:
            i -= 1
            if span[i] == UInt8(10) or span[i] == UInt8(13):
                var j = i
                while j < ws_end and (
                    span[j] == UInt8(10) or span[j] == UInt8(13)
                ):
                    j += 1
                return j - pos
        return 0

    @staticmethod
    @always_inline
    def _best_match[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:
        """Try all matchers left-to-right; return the first match.

        Uses cl100k_base (8 alt) or o200k_base (7 alt) depending on
        the ByteMapping parameter — selected at compile time.
        """
        comptime if Self.mapping == ByteMapping.SHUFFLED:
            # o200k_base: 7 alternatives
            var m = GPT4Pretokenizer._match_o200k_alt1(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_o200k_alt2(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_digit_run(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_o200k_punct_run(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_o200k_newline(span, pos)
            if m > 0:
                return m
            m = Self.match_ws_not_before_nonws(span, pos)
            if m > 0:
                return m
            return GPT4Pretokenizer._match_o200k_ws_run(span, pos)
        else:
            # cl100k_base: 8 alternatives
            var m = GPT4Pretokenizer._match_contraction(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_letter_run(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_digit_run(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_punct_run(span, pos)
            if m > 0:
                return m
            m = Self.match_trailing_all_ws(span, pos)
            if m > 0:
                return m
            m = GPT4Pretokenizer._match_newline(span, pos)
            if m > 0:
                return m
            m = Self.match_ws_not_before_nonws(span, pos)
            if m > 0:
                return m
            return Self.match_single_ws(span, pos)

    def split_view[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin]) raises -> List[StringSlice[origin]]:
        """Split text into zero-copy word views (GPT-4 cl100k_base).

        No per-word heap allocation — each entry is a view into the
        original ``text`` input.
        """
        var result = List[StringSlice[origin]]()
        var n = text.byte_length()
        if n == 0:
            return result^
        var span = text.as_bytes()
        var pos = 0
        while pos < n:
            var best_len = Self._best_match(span, pos)
            if best_len == 0:
                best_len = utf8_byte_length(span[pos])
            var byte_span = span[pos : pos + best_len]
            result.append(StringSlice(unsafe_from_utf8=byte_span))
            pos += best_len
        return result^

    def count_words[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin], mut counts: WordCounts) raises:
        """Fused word counting: same matcher loop as split_view, but each
        matched span goes straight into ``counts`` — no StringSlice list,
        no String materialization.
        """
        var n = text.byte_length()
        if n == 0:
            return
        var span = text.as_bytes()
        var sp = span.unsafe_ptr()
        var pos = 0
        while pos < n:
            var best_len = Self._best_match(span, pos)
            if best_len == 0:
                best_len = utf8_byte_length(sp[pos])
            counts.add(sp + pos, best_len)
            pos += best_len

    def write_to[T: Writer](self, mut writer: T):
        writer.write(String("GPT4Pretokenizer"))
