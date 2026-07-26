"""
Pure-Mojo pre-tokenizers: GPT-2 / GPT-4 / Ġ convention.

Three implementations sharing a common trait + UTF-8 helpers.
"""

# ===========================================================================
# Shared UTF-8 helpers
# ===========================================================================

@always_inline
def _utf8_byte_length(lead: UInt8) -> Int:
    if lead < 0x80:
        return 1
    if lead < 0xE0:
        return 2
    if lead < 0xF0:
        return 3
    return 4


@always_inline
def _codepoint_at(ptr: UnsafePointer[UInt8, _], length: Int) -> Int:
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
    return _is_letter(cp) or _is_digit(cp)


@always_inline
def _is_whitespace(cp: Int) -> Bool:
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


# ===========================================================================
# Trait
# ===========================================================================

trait PreTokenizer(Movable & Defaultable & ImplicitlyDeletable):
    def split(self, text: String) raises -> List[String]:
        ...


# ===========================================================================
# GPreTokenizer — Ġ convention (current implementation)
# ===========================================================================

struct GPreTokenizer(PreTokenizer):
    def __init__(out self):
        pass

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

    def split(self, text: String) raises -> List[String]:
        return Self.tokenize(text)


# ===========================================================================
# GPT-2 / r50k_base Pre-tokenizer
# ===========================================================================
#
# Pattern: '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++|
#           ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
#
# 7 matchers, tried at each position, longest wins.
# ===========================================================================

struct GPT2Pretokenizer(PreTokenizer):
    def __init__(out self):
        pass

    @staticmethod
    @always_inline
    def _match_contraction(span: Span[UInt8, _], pos: Int) -> Int:
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
        var n = len(span)
        var i = pos
        var count = 0
        while i < n:
            var b = Int(span[i])
            if b == 32 or b == 9 or b == 11 or b == 12 or b == 13 or b == 10:
                count += 1
                i += 1
            else:
                break
        if count == 0:
            return 0
        if i >= n:
            return count
        var next_b = Int(span[i])
        if next_b == 32 or next_b == 9 or next_b == 11 or next_b == 12 or next_b == 13 or next_b == 10:
            return count
        return 0

    @staticmethod
    @always_inline
    def _match_single_ws(span: Span[UInt8, _], pos: Int) -> Int:
        if pos >= len(span):
            return 0
        var b = Int(span[pos])
        if b == 32 or b == 9 or b == 11 or b == 12 or b == 13 or b == 10:
            return 1
        return 0

    @staticmethod
    def _best_match(span: Span[UInt8, _], pos: Int) raises -> Int:
        var best_len = 0
        var m = GPT2Pretokenizer._match_contraction(span, pos)
        if m > best_len:
            best_len = m
        m = GPT2Pretokenizer._match_letter_run(span, pos)
        if m > best_len:
            best_len = m
        m = GPT2Pretokenizer._match_digit_run(span, pos)
        if m > best_len:
            best_len = m
        m = GPT2Pretokenizer._match_punct_run(span, pos)
        if m > best_len:
            best_len = m
        m = GPT2Pretokenizer._match_trailing_ws(span, pos)
        if m > best_len:
            best_len = m
        m = GPT2Pretokenizer._match_ws_not_before_nonws(span, pos)
        if m > best_len:
            best_len = m
        m = GPT2Pretokenizer._match_single_ws(span, pos)
        if m > best_len:
            best_len = m
        return best_len

    def split(self, text: String) raises -> List[String]:
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
# GPT-4 / cl100k_base Pre-tokenizer (lifted from ../bpe.mojo)
# ===========================================================================
#
# Pattern:   '(?i:[sdmt]|ll|ve|re)
#           | [^\r\n\p{L}\p{N}]?+\p{L}+
#           | \p{N}{1,3}+
#           | ?[^\s\p{L}\p{N}]++[\r\n]*+
#           | \s++$
#           | \s*[\r\n]
#           | \s+(?!\S)
#           | \s
# ===========================================================================

struct GPT4Pretokenizer(PreTokenizer):
    def __init__(out self):
        pass

    @staticmethod
    @always_inline
    def _match_contraction(span: Span[UInt8, _], pos: Int) -> Int:
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
        if i < n and span[i] == UInt8(13):
            i += 1
            if i < n and span[i] == UInt8(10):
                i += 1
        elif i < n and span[i] == UInt8(10):
            i += 1
        return i - pos

    @staticmethod
    @always_inline
    def _match_newline(span: Span[UInt8, _], pos: Int) -> Int:
        var n = len(span)
        var i = pos
        if i >= n:
            return 0
        while i < n:
            var b = Int(span[i])
            if b == 32 or b == 9 or b == 11 or b == 12:
                i += 1
            else:
                break
        if i < n and (span[i] == UInt8(13) or span[i] == UInt8(10)):
            i += 1
            if i < n and span[i - 1] == UInt8(13) and span[i] == UInt8(10):
                i += 1
            return i - pos
        return 0

    @staticmethod
    @always_inline
    def _match_whitespace(span: Span[UInt8, _], pos: Int) -> Int:
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
    def _best_match(span: Span[UInt8, _], pos: Int) raises -> Int:
        var best_len = 0
        var m = GPT4Pretokenizer._match_contraction(span, pos)
        if m > best_len:
            best_len = m
        m = GPT4Pretokenizer._match_letter_run(span, pos)
        if m > best_len:
            best_len = m
        m = GPT4Pretokenizer._match_digit_run(span, pos)
        if m > best_len:
            best_len = m
        m = GPT4Pretokenizer._match_punct_run(span, pos)
        if m > best_len:
            best_len = m
        m = GPT4Pretokenizer._match_newline(span, pos)
        if m > best_len:
            best_len = m
        m = GPT4Pretokenizer._match_whitespace(span, pos)
        if m > best_len:
            best_len = m
        return best_len

    def split(self, text: String) raises -> List[String]:
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
