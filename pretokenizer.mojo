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

  - GPreTokenizer (legacy, ours) -- a simpler "G convention": spaces are
    replaced with the Unicode character G (U+0120) before splitting on ASCII
    space.  This matches the original GPT-2 Python reference code.

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
                          ^              ^           ^
                          |              |           |
               +----------+     +--------+     +----+
               |                |              |
    +----------+---------+  +--+----------+  +-+--------------+
    |   GPreTokenizer    |  |GPT2PreTok.  |  |GPT4PreTok.   |
    |   (G convention)   |  |(r50k_base)  |  |(cl100k_base) |
    +--------------------+  +-------------+  +---------------+

Each struct implements the PreTokenizer trait and provides a split() method.
The choice of which implementation to use is selected at compile time via the
PT parameter of BPETokenizer[PT: PreTokenizer = GPreTokenizer].

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
  GPreTokenizer                        | G convention (legacy default)
  GPT2Pretokenizer (r50k_base)         | 7-matcher regex (GPT-2)
  GPT4Pretokenizer (cl100k_base)       | 8-matcher regex (GPT-4)
"""

# ===========================================================================
# Shared UTF-8 helpers
# ===========================================================================
#
# All three pre-tokenizers operate on raw UTF-8 byte spans
# (Span[UInt8]).  These helpers abstract away the UTF-8 decoding so
# that the matchers can work at the codepoint level.
#
# The helpers are marked @always_inline because they're called from
# hot loops -- inlining eliminates function-call overhead.

@always_inline
def _utf8_byte_length(lead: UInt8) -> Int:
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
def _codepoint_at(ptr: UnsafePointer[UInt8, _], length: Int) -> Int:
    """Decode a single Unicode codepoint from raw UTF-8 bytes.

    Reads `length` bytes from `ptr` and reconstructs the codepoint.
    No validation is performed (assumes valid UTF-8).

    Args:
        ptr: Pointer to the first byte of a UTF-8 encoding.
        length: Number of bytes (1-4, from _utf8_byte_length).

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
    return ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)


@always_inline
def _is_letter(cp: Int) -> Bool:
    """Return True if cp is a Unicode letter (like \\p{L} in regex).

    Covers the most common scripts used in English, European, and CJK text:
    Latin (including extended), Greek, Cyrillic, Arabic, Hebrew, Devanagari,
    Thai, CJK Unified Ideographs, Hangul, and more.

    This is NOT a complete Unicode General Category L match -- it only covers
    the codepoint ranges that appear in practice for tokeniser training on
    Wikipedia-scale English text.  The ranges match what the GPT-2/4 regex
    engines would match for these scripts.
    """
    if 65 <= cp <= 90:
        return True
    if 97 <= cp <= 122:
        return True
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
def _is_digit(cp: Int) -> Bool:
    """Return True if cp is a Unicode digit (like \\p{N} in regex).

    Covers ASCII digits plus common digit scripts: Arabic-Indic, Devanagari,
    Bengali, Thai, Lao, Tibetan, and others.

    Like _is_letter, this is NOT a complete \\p{N} match but covers the
    digit ranges that appear in practice for English-heavy corpora.
    """
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
def _is_letter_or_digit(cp: Int) -> Bool:
    """Return True if cp is a letter or digit.

    Equivalent to _is_letter(cp) or _is_digit(cp).
    Used in character classes like [^...\\p{L}\\p{N}] in the regex patterns.
    """
    return _is_letter(cp) or _is_digit(cp)


@always_inline
def _is_whitespace(cp: Int) -> Bool:
    """Return True if cp is a Unicode whitespace codepoint.

    Covers: tab, newline, carriage return, space, plus non-break space,
    various fixed-width spaces, line/paragraph separator, and more.

    Used by the punct-run matchers to detect the boundary where a
    punctuation token ends and whitespace begins.

    Note: GPT-2's regex uses \\s which matches ONLY ASCII whitespace
    in the Rust regex crate.  We use the broader Unicode definition
    here, but in practice the only context where it matters is the
    punct-run stop condition -- and non-ASCII whitespace before
    punctuation is vanishingly rare in English text.
    """
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


@always_inline
def _is_ascii_ws_byte(b: Int) -> Bool:
    """Return True if byte b is an ASCII whitespace byte.

    Matches: \\t (0x09), \\n (0x0A), \\v (0x0B), \\f (0x0C),
    \\r (0x0D), space (0x20).

    This is the byte-level version of \\s from the regex patterns.
    We use byte checks (not codepoint checks) because:
      - GPT-2/GPT-4 regex \\s matches ONLY ASCII whitespace bytes
      - The _match_ws_* and _match_newline matchers operate on raw
        bytes, not decoded codepoints
      - Byte-level comparisons are faster and avoid UTF-8 decoding
        overhead

    This helper exists because the same ASCII-whitespace check is used
    by multiple matchers in both GPT2Pretokenizer and GPT4Pretokenizer.
    """
    return (
        b == 0x0009
        or b == 0x000A
        or b == 0x000B
        or b == 0x000C
        or b == 0x000D
        or b == 0x0020
    )


# ===========================================================================
# PreTokenizer trait
# ===========================================================================
#
# The trait that all pre-tokenizers must implement.  BPETokenizer[PT] is
# parameterised on a type that satisfies this trait.
#
# Only one method is required: split(text: String) -> List[String].

trait PreTokenizer(Movable & Defaultable & ImplicitlyDeletable):
    """Split raw text into "words" for BPE training and encoding.

    Any struct implementing this trait can be used as the PT parameter
    of BPETokenizer[PT: PreTokenizer].  The split method is called
    during both train() and encode() to divide the input text into the
    atomic units that the BPE merge algorithm operates on.

    Requirements:
      - Movable (can be transferred with ^)
      - Defaultable (can be default-constructed)
      - ImplicitlyDeletable (drop is trivial -- no manual cleanup)

    An implementation must provide:

        def split(self, text: String) raises -> List[String]:
            ...

    The returned list contains UTF-8 string slices, one per "word".
    Each word will be encoded as a sequence of byte-level token IDs
    (0-255) before BPE merge rules are applied.
    """

    def split(self, text: String) raises -> List[String]:
        """Split text into words according to this pre-tokenizer's rules.

        Args:
            text: The raw input string (valid UTF-8).

        Returns:
            A list of word strings.  Concatenating them in order
            reconstructs the original text (the split is a partition
            of the input byte span).
        """
        ...


# ===========================================================================
# GPreTokenizer -- G convention (legacy, backward-compatible default)
# ===========================================================================
#
# This pre-tokenizer replicates the approach from OpenAI's GPT-2 Python
# reference implementation (encoder.py).  It uses the Unicode character
# G (Latin capital letter G with breve, U+0120) as a spacer marker:
#
#   1. Replace every ASCII space (0x20) with " G"  (space + spacer)
#   2. Replace every "." with " ."  (separate period from surrounding)
#   3. Split on ASCII space
#
# This ensures that word-boundary spaces are preserved as the first
# character of each word token (the G prefix), which the decoder then
# reverses by replacing G back to 0x20.
#
# Why it exists: this was the original pre-tokenizer, matching the
# simple Python-level GPT-2 reference.  It is still the default for
# backward compatibility, but new development should use
# GPT2Pretokenizer or GPT4Pretokenizer for accuracy.

struct GPreTokenizer(PreTokenizer):
    def __init__(out self):
        pass

    @staticmethod
    def tokenize[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
        spacer: StaticString = "\u0120",
    ](text: StringSlice[origin]) raises -> List[String]:
        """Apply the G-convention split (static method version).

        This is the legacy interface -- called with explicit parameters
        during early development.  The trait-aware split method below
        is the modern entry point.

        Args:
            text: Input string or string slice.
            spacer: The spacer character (default U+0120).

        Returns:
            List of word strings, each starting with G if it was
            preceded by an ASCII space in the original text.
        """
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

    def split(self, text: String) raises -> List[String]:
        """Split using the G convention (trait interface).

        Args:
            text: The raw input string.

        Returns:
            List of word strings.
        """
        return Self.tokenize(text)


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
    def __init__(out self):
        pass

    @staticmethod
    @always_inline
    def _match_contraction(span: Span[UInt8, _], pos: Int) -> Int:
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
    def _match_letter_run(span: Span[UInt8, _], pos: Int) raises -> Int:
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
        while i < n:
            var lead = span[i]
            var cplen = _utf8_byte_length(lead)
            var cp = _codepoint_at(span.unsafe_ptr() + i, cplen)
            if _is_letter(cp):
                found = True
                i += cplen
            else:
                break
        if not found:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_digit_run(span: Span[UInt8, _], pos: Int) raises -> Int:
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
            var lead = span[i]
            var cplen = _utf8_byte_length(lead)
            var cp = _codepoint_at(span.unsafe_ptr() + i, cplen)
            if _is_digit(cp):
                found = True
                i += cplen
            else:
                break
        if not found:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_punct_run(span: Span[UInt8, _], pos: Int) raises -> Int:
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
            var lead = span[i]
            var cplen = _utf8_byte_length(lead)
            var cp = _codepoint_at(span.unsafe_ptr() + i, cplen)
            if _is_whitespace(cp) or _is_letter_or_digit(cp):
                break
            found = True
            i += cplen
        if not found:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_trailing_ws(span: Span[UInt8, _], pos: Int) -> Int:
        """Match alternative 5: \\s++$.

        Matches if ALL remaining bytes from pos to end-of-string are
        ASCII whitespace.  This corresponds to "whitespace at EOF" --
        the match must reach the end of input.

        Returns the match length (entire remaining span) or 0 if any
        non-whitespace byte follows.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        var count = 0
        for i in range(pos, n):
            if not _is_ascii_ws_byte(Int(span[i])):
                return 0
            count += 1
        return count

    @staticmethod
    @always_inline
    def _match_ws_not_before_nonws(span: Span[UInt8, _], pos: Int) -> Int:
        """Match alternative 6: \\s+(?!\\S).

        Matches a whitespace run where the byte immediately after the
        match is ALSO whitespace (or end-of-string).  This approximates
        the regex negative lookahead (?!\\S) -- match only if the next
        character is NOT non-whitespace.

        Algorithm:
          1. Scan the full ASCII-whitespace run starting at pos
          2. Try the full run as a match
          3. If the byte after the match is non-whitespace, shrink the
             match from the right until the post-match byte IS
             whitespace or EOS
          4. If even a single-byte match fails, return 0

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        if pos >= n:
            return 0
        if not _is_ascii_ws_byte(Int(span[pos])):
            return 0
        var end = pos
        while end < n and _is_ascii_ws_byte(Int(span[end])):
            end += 1
        var total = end - pos
        var ws_len = total
        while ws_len >= 1:
            var next_pos = pos + ws_len
            if next_pos >= n or _is_ascii_ws_byte(Int(span[next_pos])):
                return ws_len
            ws_len -= 1
        return 0

    @staticmethod
    @always_inline
    def _match_single_ws(span: Span[UInt8, _], pos: Int) -> Int:
        """Match alternative 7: \\s.

        Matches a single ASCII whitespace byte.  This is the catch-all
        fallback for whitespace bytes not matched by alternatives 5 or 6.

        Returns 1 if the byte at pos is ASCII whitespace, else 0.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            1 if whitespace, 0 otherwise.
        """
        if pos >= len(span):
            return 0
        if _is_ascii_ws_byte(Int(span[pos])):
            return 1
        return 0

    @staticmethod
    def _best_match(span: Span[UInt8, _], pos: Int) raises -> Int:
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
            consuming a single codepoint via _utf8_byte_length().
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
        m = GPT2Pretokenizer._match_trailing_ws(span, pos)
        if m > 0:
            return m
        m = GPT2Pretokenizer._match_ws_not_before_nonws(span, pos)
        if m > 0:
            return m
        return GPT2Pretokenizer._match_single_ws(span, pos)

    def split(self, text: String) raises -> List[String]:
        """Split text into words using the GPT-2 r50k_base pattern.

        Iterates through the UTF-8 byte span left-to-right.  At each
        position, tries all 7 matchers via _best_match().  The first
        matcher that returns a positive length wins.  If no matcher
        matches (should not happen for valid UTF-8 text), falls back
        to consuming a single codepoint.

        Args:
            text: The raw input string (valid UTF-8).

        Returns:
            A list of word strings whose concatenation reconstructs
            the original text.
        """
        var result = List[String]()
        var n = text.byte_length()
        if n == 0:
            return result^
        var span = text.as_bytes()
        var pos = 0
        while pos < n:
            var best_len = GPT2Pretokenizer._best_match(span, pos)
            if best_len == 0:
                best_len = _utf8_byte_length(span[pos])
            var byte_span = span[pos:pos + best_len]
            result.append(String(from_utf8_lossy=byte_span))
            pos += best_len
        return result^


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

struct GPT4Pretokenizer(PreTokenizer):
    def __init__(out self):
        pass

    @staticmethod
    @always_inline
    def _match_contraction(span: Span[UInt8, _], pos: Int) -> Int:
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
    def _match_letter_run(span: Span[UInt8, _], pos: Int) raises -> Int:
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
        var cplen = _utf8_byte_length(lead)
        var cp = _codepoint_at(span.unsafe_ptr() + i, cplen)
        if cp != 0x000A and cp != 0x000D and not _is_letter_or_digit(cp):
            i += cplen
        var found_letters = False
        while i < n:
            var cur_lead = span[i]
            var cur_len = _utf8_byte_length(cur_lead)
            var cur_cp = _codepoint_at(span.unsafe_ptr() + i, cur_len)
            if _is_letter(cur_cp):
                found_letters = True
                i += cur_len
            else:
                break
        if not found_letters:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_digit_run(span: Span[UInt8, _], pos: Int) raises -> Int:
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
            var lead = span[i]
            var cplen = _utf8_byte_length(lead)
            var cp = _codepoint_at(span.unsafe_ptr() + i, cplen)
            if _is_digit(cp):
                count += 1
                i += cplen
            else:
                break
        if count == 0:
            return 0
        return i - pos

    @staticmethod
    @always_inline
    def _match_punct_run(span: Span[UInt8, _], pos: Int) raises -> Int:
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
            var lead = span[i]
            var cplen = _utf8_byte_length(lead)
            var cp = _codepoint_at(span.unsafe_ptr() + i, cplen)
            if _is_whitespace(cp) or _is_letter_or_digit(cp):
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
    def _match_newline(span: Span[UInt8, _], pos: Int) -> Int:
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
        while end < n and _is_ascii_ws_byte(Int(span[end])):
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

    @staticmethod
    @always_inline
    def _match_whitespace(span: Span[UInt8, _], pos: Int) -> Int:
        """Match a non-newline ASCII whitespace run.

        Matches ASCII whitespace bytes that are NOT CR or LF:
        space (0x20), tab (0x09), vtab (0x0B), formfeed (0x0C).

        This is used internally by _match_trailing_all_ws to handle
        the case where we need to distinguish non-newline ws from
        the newline matcher.

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
        var count = 0
        while i < n:
            var b = Int(span[i])
            if b == 32 or b == 9 or b == 11 or b == 12:
                count += 1
                i += 1
            else:
                break
        if count == 0:
            return 0
        return count

    @staticmethod
    @always_inline
    def _match_trailing_all_ws(span: Span[UInt8, _], pos: Int) -> Int:
        """Match alternative 5: \\s++$.

        Like GPT-2 alt 5: matches only if ALL bytes from pos to EOF
        are ASCII whitespace (including CR and LF).  If any non-ws
        byte follows, returns 0.

        This is separated from _match_whitespace because it checks
        the "everything to EOF is ws" condition, not just "next
        character is ws".

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
            var b = Int(span[i])
            if b == 32 or b == 9 or b == 11 or b == 12 or b == 13 or b == 10:
                count += 1
                i += 1
            else:
                return 0
        return count

    @staticmethod
    @always_inline
    def _match_ws_not_before_nonws(span: Span[UInt8, _], pos: Int) -> Int:
        """Match alternative 7: \\s+(?!\\S).

        Same semantics as GPT-2 alt 6: matches a whitespace run where
        the byte after the match is also whitespace or EOS.

        The byte check includes CR and LF in addition to the other
        ASCII ws bytes (same as GPT-2's version via _is_ascii_ws_byte).

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            Number of bytes matched, or 0 if no match.
        """
        var n = len(span)
        if pos >= n:
            return 0
        var b = Int(span[pos])
        if not (b == 32 or b == 9 or b == 11 or b == 12 or b == 13 or b == 10):
            return 0
        var end = pos
        while end < n:
            var bb = Int(span[end])
            if bb == 32 or bb == 9 or bb == 11 or bb == 12 or bb == 13 or bb == 10:
                end += 1
            else:
                break
        var total = end - pos
        var ws_len = total
        while ws_len >= 1:
            var next_pos = pos + ws_len
            if next_pos >= n:
                return ws_len
            var nb = Int(span[next_pos])
            if nb == 32 or nb == 9 or nb == 11 or nb == 12 or nb == 13 or nb == 10:
                return ws_len
            ws_len -= 1
        return 0

    @staticmethod
    @always_inline
    def _match_single_ws(span: Span[UInt8, _], pos: Int) -> Int:
        """Match alternative 8: \\s.

        Catch-all for any single ASCII whitespace byte not matched by
        earlier alternatives.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            1 if whitespace, 0 otherwise.
        """
        if pos >= len(span):
            return 0
        var b = Int(span[pos])
        if b == 32 or b == 9 or b == 11 or b == 12 or b == 13 or b == 10:
            return 1
        return 0

    @staticmethod
    def _best_match(span: Span[UInt8, _], pos: Int) raises -> Int:
        """Try all 8 matchers left-to-right; return the first match.

        Implements regex alternation for the cl100k_base pattern.
        Matchers are tried in order (alternatives 1-8).  The FIRST
        one that returns a positive match length wins.

        Args:
            span: The full UTF-8 byte span of the input text.
            pos: Current position in the span.

        Returns:
            The match length in bytes, or 0 if no matcher matched.
            When 0 is returned, the caller (split()) falls back to
            consuming a single codepoint via _utf8_byte_length().
        """
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
        m = GPT4Pretokenizer._match_trailing_all_ws(span, pos)
        if m > 0:
            return m
        m = GPT4Pretokenizer._match_newline(span, pos)
        if m > 0:
            return m
        m = GPT4Pretokenizer._match_ws_not_before_nonws(span, pos)
        if m > 0:
            return m
        return GPT4Pretokenizer._match_single_ws(span, pos)

    def split(self, text: String) raises -> List[String]:
        """Split text into words using the GPT-4 cl100k_base pattern.

        Iterates through the UTF-8 byte span left-to-right.  At each
        position, tries all 8 matchers via _best_match().  The first
        matcher that returns a positive length wins.  If no matcher
        matches (should not happen for valid UTF-8 text), falls back
        to consuming a single codepoint.

        Args:
            text: The raw input string (valid UTF-8).

        Returns:
            A list of word strings whose concatenation reconstructs
            the original text.
        """
        var result = List[String]()
        var n = text.byte_length()
        if n == 0:
            return result^
        var span = text.as_bytes()
        var pos = 0
        while pos < n:
            var best_len = GPT4Pretokenizer._best_match(span, pos)
            if best_len == 0:
                best_len = _utf8_byte_length(span[pos])
            var byte_span = span[pos:pos + best_len]
            result.append(String(from_utf8_lossy=byte_span))
            pos += best_len
        return result^
