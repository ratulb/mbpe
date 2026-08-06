from std.bit import pop_count
from std.memory import memcmp
from bpe.unicode_tables import (
    is_letter,
    is_digit,
    is_letter_or_digit,
    is_upper_like,
    is_lower_like,
    is_whitespace,
)

@always_inline
def utf8_byte_length(lead: UInt8) -> Int:

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

comptime BC_LETTER = 1
comptime BC_DIGIT = 2
comptime BC_WS = 4
comptime BC_CRLF = 8

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

@always_inline
def _hasless(x: UInt64, n: UInt64) -> UInt64:
    return (
        (x - n * UInt64(0x0101010101010101)) & ~x & UInt64(0x8080808080808080)
    )

@always_inline
def _haszero(x: UInt64) -> UInt64:
    return (
        (x - UInt64(0x0101010101010101)) & ~x & UInt64(0x8080808080808080)
    )

@always_inline
def _letters8(x: UInt64) -> UInt64:
    var l = x | UInt64(0x2020202020202020)
    var below_z = _hasless(l, UInt64(0x7B))

    var is_0x7B = _haszero(l ^ (UInt64(0x7B) * UInt64(0x0101010101010101)))
    below_z = below_z & ~is_0x7B
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

    var p8 = span.unsafe_ptr()
    var consumed = 0
    var j = i
    while j + 8 <= n:
        var w: UInt64 = (p8 + j).bitcast[UInt64]()[]
        var nl = ~_letters8(w) & UInt64(0x8080808080808080)
        if nl != 0:
            var lsb = nl & (UInt64(0) - nl)

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

    return (Int(BYTE_CLASS[b]) & BC_WS) != 0

@always_inline
def is_ws_at[
    origin: Origin, //
](span: Span[UInt8, origin], pos: Int) -> Int:

    var lead = span[pos]
    if lead < 0x80:
        if is_ascii_ws_byte(Int(lead)):
            return 1
        return 0
    var cplen = utf8_byte_length(lead)
    if is_whitespace(decode_codepoint(span.unsafe_ptr() + pos, cplen)):
        return cplen
    return 0

@fieldwise_init
struct ByteMapping(ImplicitlyCopyable & Equatable):

    var _value: Int

    comptime SEQUENTIAL = ByteMapping(0)
    comptime SHUFFLED = ByteMapping(1)

comptime O200K_BYTE_TO_ID = SIMD[DType.int32, 256](
    188, 189, 190, 191, 192, 193, 194, 195,
    196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 206, 207, 208, 209, 210, 211,
    212, 213, 214, 215, 216, 217, 218, 219,
    220,
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93,
    221,
    222, 223, 224, 225, 226, 227, 228, 229,
    230, 231, 232, 233, 234, 235, 236, 237,
    238, 239, 240, 241, 242, 243, 244, 245,
    246, 247, 248, 249, 250, 251, 252, 253,
    254,
    94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105,
    255,
    106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137,
    138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161,
    162, 163, 164, 165, 166, 167, 168, 169,
    170, 171, 172, 173, 174, 175, 176, 177,
    178, 179, 180, 181, 182, 183, 184, 185,
    186, 187,
)

comptime O200K_ID_TO_BYTE = SIMD[DType.int32, 256](
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
    0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40,
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,
    0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60,
    0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70,
    0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E,
    0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8,
    0xA9, 0xAA, 0xAB, 0xAC, 0xAE, 0xAF, 0xB0, 0xB1,
    0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9,
    0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF, 0xC0, 0xC1,
    0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
    0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1,
    0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
    0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0, 0xE1,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1,
    0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9,
    0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    0x20,
    0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E,
    0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E,
    0x9F, 0xA0, 0xAD,
)

from bpe.shared import IntArray, ByteArray, TokenSpan, ByteSpanArena

struct WordCounts(ImplicitlyCopyable & Movable):

    comptime FNV_offset_basis = UInt64(0xCBF29CE484222325)

    comptime FNV_prime = UInt64(0x100000001B3)

    var arena: ByteSpanArena
    var slots: IntArray
    var counts: IntArray
    var n_entries: Int
    var slot_cap: Int

    def __init__(out self, capacity: Int = 4096):

        var cap = 16
        while cap < capacity * 4:
            cap *= 2
        self.slot_cap = cap
        self.slots = IntArray(length=cap, fill=0)
        self.arena = ByteSpanArena()
        self.arena.spans.reserve(capacity)
        self.counts = IntArray(capacity=capacity)
        self.n_entries = 0

    def __init__(out self, *, copy: Self):

        self.slot_cap = copy.slot_cap
        self.slots = copy.slots.copy()
        self.arena = ByteSpanArena(copy=copy.arena)
        self.counts = copy.counts.copy()
        self.n_entries = copy.n_entries

    def __init__(out self, *, deinit move: Self):

        self.slot_cap = move.slot_cap
        self.slots = move.slots^
        self.arena = move.arena^
        self.counts = move.counts^
        self.n_entries = move.n_entries

    def _rehash(mut self, new_cap: Int):

        self.slots = IntArray(length=new_cap, fill=0)
        self.slot_cap = new_cap
        var nsp = self.slots.unsafe_ptr()
        var bp = self.arena.bytes.unsafe_ptr()

        for e in range(self.n_entries):
            var h = Self._fnv1a64(
                bp + self.arena.spans[e].offset, self.arena.spans[e].length
            )
            var idx = Int(h & UInt64(new_cap - 1))

            while nsp[idx] != 0:
                idx = (idx + 1) & (new_cap - 1)
            nsp[idx] = e + 1

    def add[
        origin: Origin, //
    ](mut self, ptr: UnsafePointer[UInt8, origin], length: Int):

        if length == 0:
            return
        if self.n_entries * 2 >= self.slot_cap:
            self._rehash(self.slot_cap * 2)
        var h = Self._fnv1a64(ptr, length)

        var slot_cap = self.slot_cap
        var sp = self.slots.unsafe_ptr()
        var spans_ptr = self.arena.spans.unsafe_ptr()
        var bytes_ptr = self.arena.bytes.unsafe_ptr().as_noalias_ptr()
        var counts_ptr = self.counts.unsafe_ptr()
        var idx = Int(h & UInt64(slot_cap - 1))
        while sp[idx] != 0:
            var e = sp[idx] - 1
            if (
                spans_ptr[e].length == length
                and memcmp(
                    bytes_ptr + spans_ptr[e].offset,
                    ptr,
                    length,
                )
                == 0
            ):

                counts_ptr[e] = counts_ptr[e] + 1
                return

            idx = (idx + 1) & (slot_cap - 1)

        var e = self.n_entries
        _ = self.arena.add(ptr, length)
        sp[idx] = e + 1
        self.counts.append(1)
        self.n_entries += 1

    def add[origin: Origin, //](mut self, span: Span[UInt8, origin]):

        self.add(span.unsafe_ptr(), len(span))

    def add[
        origin: Origin, //
    ](mut self, start: UnsafePointer[UInt8, origin], pos: Int, length: Int):

        self.add(start + pos, length)

    @staticmethod
    @always_inline
    def _fnv1a64[
        origin: Origin, //
    ](ptr: UnsafePointer[UInt8, origin], n: Int) -> UInt64:

        var h: UInt64 = Self.FNV_offset_basis
        for i in range(n):
            h ^= UInt64(ptr[i])
            h *= Self.FNV_prime
        return h

trait PreTokenizer(Movable & Defaultable & ImplicitlyDeletable & Writable):

    comptime byte_map: ByteMapping

    @staticmethod
    @always_inline
    def byte_to_id(b: Int) -> Int:

        return b

    @staticmethod
    @always_inline
    def id_to_byte(rank: Int) -> Int:

        return rank

    def split_view[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin]) raises -> List[StringSlice[origin]]:

        var result = List[StringSlice[origin]]()
        result.reserve(1)
        result.append(text)
        return result^

    def split[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin]) raises -> List[String]:

        var views = self.split_view(text)
        var result = List[String](capacity=len(views))
        for v in views:
            result.append(String(v))
        return result^

    def count_words[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSlice[origin], mut counts: WordCounts) raises:

        ref views = self.split_view(text)
        for v in views:
            counts.add(v.as_bytes())

    @staticmethod
    def name() -> String:
        ...

    @staticmethod
    def special_tokens() -> Dict[String, Int]:
        ...

    @staticmethod
    def match_trailing_all_ws[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

        var n = len(span)
        var i = pos
        var count = 0
        while i < n:
            var l = is_ws_at(span, i)
            if l == 0:
                return 0
            count += l
            i += l
        return count

    @staticmethod
    def match_ws_not_before_nonws[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

        var n = len(span)
        if pos >= n:
            return 0
        if is_ws_at(span, pos) == 0:
            return 0
        var end = pos
        var total = 0
        var last_cp_len = 0
        while end < n:
            var l = is_ws_at(span, end)
            if l == 0:
                break
            last_cp_len = l
            total += l
            end += l
        if end < n:
            return total - last_cp_len
        return total

    @staticmethod
    def match_single_ws[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

        if pos >= len(span):
            return 0
        return is_ws_at(span, pos)

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

        comptime if Self.mapping == ByteMapping.SHUFFLED:
            return Int(O200K_BYTE_TO_ID[b])
        else:
            return b

    @staticmethod
    def id_to_byte(rank: Int) -> Int:

        comptime if Self.mapping == ByteMapping.SHUFFLED:
            return Int(O200K_ID_TO_BYTE[rank])
        else:
            return rank

    @staticmethod
    @always_inline
    def _match_contraction[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

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

        var n = len(span)
        if pos >= n:
            return 0
        var i = pos
        var last_crlf_end = -1
        while i < n:
            var l = is_ws_at(span, i)
            if l == 0:
                break
            if span[i] == UInt8(10) or span[i] == UInt8(13):
                last_crlf_end = i + 1
            i += l
        if last_crlf_end < 0:
            return 0
        return last_crlf_end - pos

    @staticmethod
    @always_inline
    def _lower_len_at[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

        var b = span[pos]
        if b < 0x80:
            var ib = Int(b)
            if 97 <= ib <= 122:
                return 1
            return 0
        var cplen = utf8_byte_length(b)
        if is_lower_like(decode_codepoint(span.unsafe_ptr() + pos, cplen)):
            return cplen
        return 0

    @staticmethod
    @always_inline
    def _upper_len_at[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

        var b = span[pos]
        if b < 0x80:
            var ib = Int(b)
            if 65 <= ib <= 90:
                return 1
            return 0
        var cplen = utf8_byte_length(b)
        if is_upper_like(decode_codepoint(span.unsafe_ptr() + pos, cplen)):
            return cplen
        return 0

    @staticmethod
    @always_inline
    def _prefix_len[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

        var lead = span[pos]
        if lead < 0x80:
            if (
                Int(BYTE_CLASS[Int(lead)]) & (BC_LETTER | BC_DIGIT | BC_CRLF)
            ) == 0:
                return 1
            return 0
        var cplen = utf8_byte_length(lead)
        var cp = decode_codepoint(span.unsafe_ptr() + pos, cplen)
        if cp != 0x000A and cp != 0x000D and not is_letter_or_digit(cp):
            return cplen
        return 0

    @staticmethod
    @always_inline
    def _alt1_from[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int, preflen: Int) raises -> Int:

        var n = len(span)
        var start = pos + preflen
        if start >= n:
            return 0
        var best_B = -1
        if GPT4Pretokenizer._lower_len_at(span, start) > 0:
            best_B = start
        var i = start
        while i < n:
            var l = GPT4Pretokenizer._upper_len_at(span, i)
            if l == 0:
                break
            i += l
            if i < n and GPT4Pretokenizer._lower_len_at(span, i) > 0:
                best_B = i
        if best_B < 0:
            return 0
        var j = best_B
        while j < n:
            var l = GPT4Pretokenizer._lower_len_at(span, j)
            if l == 0:
                break
            j += l
        if j == best_B:
            return 0
        var cont = GPT4Pretokenizer._match_contraction(span, j)
        if cont > 0:
            j += cont
        return j - pos

    @staticmethod
    @always_inline
    def _alt2_from[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int, preflen: Int) raises -> Int:

        var n = len(span)
        var i = pos + preflen
        if i >= n:
            return 0
        var upper_start = i
        while i < n:
            var l = GPT4Pretokenizer._upper_len_at(span, i)
            if l == 0:
                break
            i += l
        if i == upper_start:
            return 0
        while i < n:
            var l = GPT4Pretokenizer._lower_len_at(span, i)
            if l == 0:
                break
            i += l
        var cont = GPT4Pretokenizer._match_contraction(span, i)
        if cont > 0:
            i += cont
        return i - pos

    @staticmethod
    @always_inline
    def _match_o200k_alt1[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:

        var n = len(span)
        if pos >= n:
            return 0
        var preflen = GPT4Pretokenizer._prefix_len(span, pos)
        var with_pref = GPT4Pretokenizer._alt1_from(span, pos, preflen)
        if with_pref > 0:
            return with_pref
        return GPT4Pretokenizer._alt1_from(span, pos, 0)

    @staticmethod
    @always_inline
    def _match_o200k_alt2[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:

        var n = len(span)
        if pos >= n:
            return 0
        var preflen = GPT4Pretokenizer._prefix_len(span, pos)
        var with_pref = GPT4Pretokenizer._alt2_from(span, pos, preflen)
        if with_pref > 0:
            return with_pref
        return GPT4Pretokenizer._alt2_from(span, pos, 0)

    @staticmethod
    @always_inline
    def _match_o200k_punct_run[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:

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

        var n = len(span)
        if pos >= n:
            return 0
        var total = 0
        var i = pos
        while i < n:
            var l = is_ws_at(span, i)
            if l == 0:
                break
            total += l
            i += l
        return total

    @staticmethod
    @always_inline
    def _match_o200k_newline[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) -> Int:

        var n = len(span)
        if pos >= n:
            return 0
        var i = pos
        var last_crlf_end = -1
        while i < n:
            var l = is_ws_at(span, i)
            if l == 0:
                break
            if span[i] == UInt8(10) or span[i] == UInt8(13):
                last_crlf_end = i + 1
            i += l
        if last_crlf_end < 0:
            return 0
        return last_crlf_end - pos

    @staticmethod
    @always_inline
    def _best_match[
        origin: Origin, //
    ](span: Span[UInt8, origin], pos: Int) raises -> Int:

        comptime if Self.mapping == ByteMapping.SHUFFLED:

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

