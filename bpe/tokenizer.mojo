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
from bpe.shared import IntArray, ByteArray, TokenSpan, ByteSpanArena

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

comptime CACHE_SHIFT: Int = 10
comptime CACHE_SIZE: Int = 1000
comptime CACHE_ENTRIES: Int = 1 << (CACHE_SHIFT * 2)
comptime ENCODE_SHIFT: Int = 20
comptime ENCODE_MASK: Int = (1 << ENCODE_SHIFT) - 1

comptime HEAP_SHIFT: Int = 24
comptime HEAP_MASK: Int = (1 << HEAP_SHIFT) - 1

comptime SCAN_LIMIT: Int = 32

@fieldwise_init
struct HeapKey(ImplicitlyCopyable & RegisterPassable):

    var rank: Int
    var node: Int

@always_inline
def _pack_heap_key(rank: Int, node: Int) -> Int:

    return -(rank << HEAP_SHIFT | node)

@always_inline
def _unpack_heap_key(key: Int) -> HeapKey:

    var raw = -key
    return HeapKey(raw >> HEAP_SHIFT, raw & HEAP_MASK)

struct MergeLookup(ImplicitlyCopyable & Movable & Writable):

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

struct TokenByteTable(ImplicitlyCopyable & Movable & Sized & Writable):
    var arena: ByteSpanArena

    def __init__(out self):
        self.arena = ByteSpanArena()

    def __init__(out self, *, copy: Self):

        self.arena = ByteSpanArena(copy=copy.arena)

    def __init__(out self, *, deinit move: Self):
        self.arena = move.arena^

    @always_inline
    def __len__(self) -> Int:
        return len(self.arena)

    def write_to[T: Writer](self, mut writer: T):

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

        _ = self.arena.add(raw)

    @always_inline
    def set_bytes[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, id: Int, raw: Span[Byte, origin]):

        while len(self.arena) <= id:
            self.arena.spans.append(TokenSpan(len(self.arena.bytes), 0))
        var off = len(self.arena.bytes)
        self.arena.bytes.reserve(off + len(raw))
        self.arena.bytes.extend(raw)
        self.arena.spans[id] = TokenSpan(off, len(raw))

@fieldwise_init
struct MergeScratch(ImplicitlyCopyable & RegisterPassable):
    var ids: UnsafePointer[Int, MutAnyOrigin]
    var nxt: UnsafePointer[Int, MutAnyOrigin]
    var prv: UnsafePointer[Int, MutAnyOrigin]
    var alive: UnsafePointer[UInt8, MutAnyOrigin]
    var cap: Int

    def __init__(out self):
        self.ids = alloc[Int](0)
        self.nxt = alloc[Int](0)
        self.prv = alloc[Int](0)
        self.alive = alloc[UInt8](0)
        self.cap = 0

    @always_inline
    def free(mut self):

        self.ids.free()
        self.nxt.free()
        self.prv.free()
        self.alive.free()
        self.cap = 0

    def ensure_capacity(mut self, n: Int):

        if n > self.cap:
            self.free()
            self.ids = alloc[Int](n)
            self.nxt = alloc[Int](n)
            self.prv = alloc[Int](n)
            self.alive = alloc[UInt8](n)
            self.cap = n

struct BPETokenizer[PT: PreTokenizer = GPT2Pretokenizer](
    Sized & Movable & Writable
):

    var pt: Self.PT

    var merges: List[MergeRule]

    var lookup_table: MergeLookup

    var byte_to_cp: Dict[Int, Int]

    var byte_to_rank: IntArray

    var token_table: TokenByteTable

    var special_bytes: Dict[String, Int]

    var inverse_special: Dict[Int, String]

    def __init__(out self):

        self.pt = Self.PT()
        self.merges = List[MergeRule]()
        self.lookup_table = MergeLookup()
        self.byte_to_cp = Dict[Int, Int]()

        self.byte_to_rank = [b for b in range(256)]
        self.token_table = TokenByteTable()
        self.special_bytes = Dict[String, Int]()
        self.inverse_special = Dict[Int, String]()

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

                var cp = 256 + n
                self.byte_to_cp[b] = cp
                n += 1

    def register_special_tokens(mut self, tokens: Dict[String, Int]) raises:

        for item in tokens.items():
            self._register_special_token(item.key, item.value)

    def _register_special_token(mut self, text: String, id: Int) raises:

        if text.byte_length() == 0:
            raise Error("special token text must not be empty")
        if text in self.special_bytes:
            raise Error("duplicate special token: " + text)
        self.special_bytes[text] = id
        self.inverse_special[id] = text
        self.token_table.set_bytes(id, text.as_bytes())

    def train[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, corpus: Span[String, origin], vocab_size: Int) raises:

        if vocab_size < 256:
            raise Error(
                "vocab_size must be at least 256 to hold the base byte"
                " vocabulary"
            )

        var word_counts = WordCounts()
        for text in corpus:
            self.pt.count_words(text, word_counts)

        self.token_table = TokenByteTable()
        self.token_table.reserve(vocab_size)
        self.byte_to_rank = [b for b in range(256)]
        for rank in range(256):

            var b = Self.PT.id_to_byte(rank)
            var raw = ByteArray(capacity=1)
            raw.append(Byte(b))
            self.token_table.add(Span[Byte](raw))

        var total_tokens: Int = 0
        for ei in range(
            word_counts.n_entries
        ):
            total_tokens += word_counts.arena.spans[ei].length
        var arena = IntArray(capacity=total_tokens)
        var word_offs = IntArray(
            capacity=word_counts.n_entries
        )
        var word_len = IntArray(
            capacity=word_counts.n_entries
        )
        var word_freq = IntArray(
            capacity=word_counts.n_entries
        )
        var stats = Dict[
            Int, Int
        ]()
        var where_dict = Dict[
            Int, IntArray
        ]()
        var wb = word_counts.arena.bytes.unsafe_ptr()
        for ei in range(
            word_counts.n_entries
        ):
            var off = word_counts.arena.spans[ei].offset
            var ln = word_counts.arena.spans[ei].length
            var freq = word_counts.counts[ei]
            var off_arena = len(arena)
            word_offs.append(off_arena)

            for i in range(ln):
                arena.append(Self.PT.byte_to_id(Int(wb[off + i])))
            word_len.append(ln)
            word_freq.append(freq)
            var iw = len(word_offs) - 1

            for i in range(off_arena, off_arena + ln - 1):
                var key = (arena[i] << ENCODE_SHIFT) | arena[i + 1]
                stats[key] = stats.get(key, 0) + freq
                if key not in where_dict:
                    where_dict[key] = IntArray()
                where_dict[key].append(iw)

        self.lookup_table = MergeLookup()
        self.merges = List[MergeRule]()
        while len(self.token_table) < vocab_size:

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

                break

            var a_id = best_pair[0]
            var b_id = best_pair[1]
            var best_key = (a_id << ENCODE_SHIFT) | b_id
            var merged_id = len(self.token_table)

            var snap = len(where_dict[best_key])
            for wi in range(snap):
                var iw = where_dict[best_key][wi]
                var freq = word_freq[iw]
                var start = word_offs[iw]
                var n = word_len[iw]
                var wt = arena.unsafe_ptr() + start

                var w = 0
                var i = 0
                while i < n:
                    if i < n - 1 and wt[i] == a_id and wt[i + 1] == b_id:

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

                        if w > 0:

                            var pk2 = (wt[w - 1] << ENCODE_SHIFT) | merged_id
                            stats[pk2] = stats.get(pk2, 0) + freq
                            if pk2 not in where_dict:
                                where_dict[pk2] = IntArray()
                            where_dict[pk2].append(iw)
                        if i + 2 < n:

                            var nk2 = (merged_id << ENCODE_SHIFT) | wt[i + 2]
                            stats[nk2] = stats.get(nk2, 0) + freq
                            if nk2 not in where_dict:
                                where_dict[nk2] = IntArray()
                            where_dict[nk2].append(iw)

                        wt[w] = merged_id
                        w += 1
                        i += 2
                    else:

                        wt[w] = wt[i]
                        w += 1
                        i += 1

                word_len[iw] = w

            self.merges.append(MergeRule(a_id, b_id, merged_id))
            self.lookup_table.set(a_id, b_id, merged_id)

            var spans = self.token_table.arena.spans.unsafe_ptr()
            var pool = (
                self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
            )
            var la = spans[a_id].length
            var lb = spans[b_id].length
            var merged_bytes = ByteArray(capacity=la + lb)
            merged_bytes.resize(la + lb, 0)
            memcpy(
                dest=merged_bytes.unsafe_ptr(),
                src=pool + spans[a_id].offset,
                count=la,
            )
            memcpy(
                dest=merged_bytes.unsafe_ptr() + la,
                src=pool + spans[b_id].offset,
                count=lb,
            )
            self.token_table.add(Span[Byte](merged_bytes))

    @always_inline
    def _copy_word_ids[origin: Origin](
        self,
        ptr: UnsafePointer[UInt8, origin],
        n: Int,
        dst: UnsafePointer[Int, MutAnyOrigin],
    ):

        var btr = self.byte_to_rank.unsafe_ptr()
        for i in range(n):
            comptime if Self.PT.byte_map == ByteMapping.SHUFFLED:
                dst[i] = Self.PT.byte_to_id(Int(ptr[i]))
            else:
                dst[i] = btr[Int(ptr[i])]

    @always_inline
    def _merge_scan[origin: Origin](
        self,
        ptr: UnsafePointer[UInt8, origin],
        n: Int,
        dst: UnsafePointer[Int, MutAnyOrigin],
    ) -> Int:

        self._copy_word_ids(ptr, n, dst)
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
        return len

    @always_inline
    def _merge_heap[origin: Origin](
        self,
        ptr: UnsafePointer[UInt8, origin],
        n: Int,
        dst: UnsafePointer[Int, MutAnyOrigin],
        mut scratch: MergeScratch,
        mut heap: BinaryHeap[Int],
    ) -> Int:

        if n > scratch.cap:

            scratch.ensure_capacity(n)

        self._copy_word_ids(ptr, n, scratch.ids)

        for i in range(n):
            scratch.nxt[i] = i + 1
            scratch.prv[i] = i - 1
            scratch.alive[i] = 1
        scratch.nxt[n - 1] = -1

        for i in range(n - 1):
            var r0 = self.lookup_table.get(scratch.ids[i], scratch.ids[i + 1])
            if r0 >= 0:
                heap.push(_pack_heap_key(r0, i))

        while len(heap) > 0:
            var key = _unpack_heap_key(heap.pop())
            var e = key.node
            var rank = key.rank
            if scratch.alive[e] == 0:

                continue
            var j = scratch.nxt[e]
            if j < 0:

                continue

            if self.lookup_table.get(scratch.ids[e], scratch.ids[j]) != rank:
                continue

            scratch.ids[e] = rank
            var k = scratch.nxt[j]
            if k >= 0:
                scratch.prv[k] = e
            scratch.nxt[e] = k
            scratch.alive[j] = 0

            var p = scratch.prv[e]
            if p >= 0:
                var rp = self.lookup_table.get(scratch.ids[p], scratch.ids[e])
                if rp >= 0:
                    heap.push(_pack_heap_key(rp, p))
            if k >= 0:
                var rk = self.lookup_table.get(scratch.ids[e], scratch.ids[k])
                if rk >= 0:
                    heap.push(_pack_heap_key(rk, e))

        var count = 0
        var cur = 0
        while cur >= 0:
            dst[count] = scratch.ids[cur]
            count += 1
            cur = scratch.nxt[cur]
        return count

    def encode_ordinary[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> IntArray:

        if text.byte_length() == 0:
            return IntArray()

        ref words = self.pt.split_view(text)

        var total_bytes = 0
        for word in words:
            total_bytes += word.byte_length()

        var result = IntArray(unsafe_uninit_length=total_bytes)
        var write_pos = 0
        var scratch = MergeScratch()
        var heap = BinaryHeap[Int]()
        for word in words:
            var ptr = word.unsafe_ptr()
            var n = word.byte_length()
            var dst = result.unsafe_ptr() + write_pos

            if n < 2:

                self._copy_word_ids(ptr, n, dst)
                write_pos += n
            elif n < SCAN_LIMIT:
                write_pos += self._merge_scan(ptr, n, dst)
            else:
                write_pos += self._merge_heap(ptr, n, dst, scratch, heap)

        scratch.free()

        result.resize(write_pos, 0)
        return result^

    def pretokenize[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> List[StringSlice[origin]]:

        return self.pt.split_view(text)

    def encode[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ](self, text: StringSlice[origin]) raises -> IntArray:

        if len(self.special_bytes) == 0:
            return self.encode_ordinary(text)

        var n = text.byte_length()
        if n == 0:
            return IntArray()

        var bytes = text.as_bytes()
        var result = IntArray()
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

                    var seg = StringSlice(
                        unsafe_from_utf8=bytes[start:next_special]
                    )
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

    def decode[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, ids: Span[Int, origin]) raises -> String:

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
            if n > 0:
                memcpy(
                    dest=dst + write_offset,
                    src=ptr + spans[id].offset,
                    count=n,
                )
                write_offset += n
        return result^

    def __len__(self) -> Int:

        return len(self.token_table)

    def name(self) -> String:

        return Self.PT.name()

    def decode_bytes[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, ids: Span[Int, origin]) raises -> ByteArray:

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

            starts.append(write_offset)
            if n > 0:
                memcpy(
                    dest=dst + write_offset, src=ptr + spans[id].offset, count=n
                )
                write_offset += n
            ends.append(write_offset)
        return result^

    def token_byte_values(self) -> List[ByteArray]:

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

        for item in self.special_bytes.items():
            if item.key == text:
                return item.value
        for i in range(len(self.token_table)):
            if self.display_of(i) == text:
                return i
        raise Error("unknown token: " + String(text))

    def write_to[T: Writer](self, mut writer: T):

        writer.write(
            String("BPETokenizer(vocab_size=")
            + String(len(self.token_table))
            + String(")")
        )

    @staticmethod
    def _bytes_key[
        mut: Bool, //, origin: Origin[mut=mut]
    ](bytes: Span[Byte, origin]) -> String:

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

        var parts = List[ByteArray](capacity=len(token_bytes))
        for i in range(len(token_bytes)):
            var single = ByteArray(capacity=1)
            single.append(token_bytes[i])
            parts.append(single^)

        while True:

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

            if min_idx < 0 or (max_rank >= 0 and min_rank >= max_rank):
                break

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

                var lk = Self._bytes_key(Span[Byte](parts[0]))
                var rk = Self._bytes_key(Span[Byte](parts[1]))
                if lk in mergeable_ranks and rk in mergeable_ranks:
                    left_id = mergeable_ranks[lk]
                    right_id = mergeable_ranks[rk]
            elif len(parts) > 2:

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

                recovered.append(MergeRule(left_id, right_id, token_id))
        self.merges = recovered^

    def save_tiktoken(mut self, path: String) raises:

        var spans = self.token_table.arena.spans.unsafe_ptr()
        var pool = self.token_table.arena.bytes.unsafe_ptr().as_noalias_ptr()
        with open(path, "w") as f:
            for token_id in range(len(self.token_table)):
                if token_id in self.inverse_special:
                    continue

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

        var file_content: String
        with open(path, "r") as f:
            file_content = f.read()
        var raw_lines = file_content.split("\n")

        var mergeable_ranks = Dict[String, Int]()
        var all_tokens = List[ByteArray]()
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
                all_tokens.append(ByteArray())
            all_tokens[rank] = raw^
            if rank > max_id:
                max_id = rank

        if len(all_tokens) == 0:
            raise Error(
                "load_tiktoken: '" + path + "' contains no token lines"
            )

        var new_vocab_size = max_id + 1
        var new_table = TokenByteTable()
        new_table.reserve(new_vocab_size)
        for token_id in range(new_vocab_size):
            new_table.add(Span[Byte](all_tokens[token_id]))

        self._recover_merges(mergeable_ranks, all_tokens)

        var new_lookup_table = MergeLookup()
        for merge in self.merges:
            new_lookup_table.set(merge.first, merge.second, merge.merged)

        var single_byte = ByteArray()
        single_byte.resize(1, 0)
        for b in range(256):
            single_byte[0] = Byte(b)
            var key = BPETokenizer._bytes_key(
                Span[Byte](ptr=single_byte.unsafe_ptr(), length=1)
            )
            if key in mergeable_ranks:
                self.byte_to_rank[b] = mergeable_ranks[key]

        self.token_table = new_table^
        self.lookup_table = new_lookup_table^

        for item in Self.PT.special_tokens().items():
            if not item.value in self.inverse_special:
                self._register_special_token(item.key, item.value)

@always_inline
def merge_inplace(
    buf: UnsafePointer[Int, MutAnyOrigin],
    n: Int,
    a: Int,
    b: Int,
    m: Int,
) -> Int:

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

def _find_data_dir() raises -> String:

    var env_dir = getenv("MBPE_DATA_DIR", "")
    if env_dir.byte_length() > 0:
        return env_dir
    return "data"

struct Tokenizers:

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

