from std.format import Writer
from std.memory import memcpy

comptime IntArray = List[Int]
comptime ByteArray = List[Byte]

@fieldwise_init
struct TokenSpan(
    ImplicitlyCopyable
    & Movable
    & TrivialRegisterPassable
    & Equatable
    & Writable
):
    """One entry's slice of a byte arena: (offset, length) into `bytes`.

    `offset` is the index of the first byte in the pool; `length` is the
    number of bytes.  Combining the two into one small struct (instead of
    two parallel IntArrays) keeps every entry's bounds in a single cache
    line on the hot paths.
    """

    var offset: Int
    var length: Int

    def write_to[T: Writer](self, mut writer: T):
        writer.write(
            String("TokenSpan(offset=")
            + String(self.offset)
            + String(", length=")
            + String(self.length)
            + String(")")
        )


struct ByteSpanArena(ImplicitlyCopyable & Movable & Sized & Writable):
    """Flat byte pool plus per-entry (offset, length) spans.

    Append-only storage: `add` copies bytes into the pool (a single memcpy
    via `bytes.extend`) and returns the new span's index.  Shared by
    TokenByteTable (per-token bytes) and WordCounts (per-word bytes).
    """

    var bytes: ByteArray
    var spans: List[TokenSpan]

    def __init__(out self):
        self.bytes = ByteArray()
        self.spans = List[TokenSpan]()

    def __init__(out self, *, copy: Self):
        """Deep copy of the byte pool and the span list."""
        self.bytes = ByteArray(copy=copy.bytes)
        self.spans = List[TokenSpan](copy=copy.spans)

    def __init__(out self, *, deinit move: Self):
        self.bytes = move.bytes^
        self.spans = move.spans^

    @always_inline
    def __len__(self) -> Int:
        return len(self.spans)

    def write_to[T: Writer](self, mut writer: T):
        writer.write(
            String("ByteSpanArena(bytes=")
            + String(len(self.bytes))
            + String(", spans=")
            + String(len(self.spans))
            + String(")")
        )

    @always_inline
    def add[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, raw: Span[Byte, origin]) -> Int:
        """Append `raw` to the pool; return the new span's index."""
        var idx = len(self.spans)
        var off = len(self.bytes)
        self.bytes.reserve(off + len(raw))
        self.bytes.extend(raw)
        self.spans.append(TokenSpan(off, len(raw)))
        return idx

    @always_inline
    def add[
        origin: Origin, //
    ](mut self, ptr: UnsafePointer[UInt8, origin], length: Int) -> Int:
        """Append `length` bytes at `ptr` to the pool; return the new span's
        index."""
        var idx = len(self.spans)
        var off = len(self.bytes)
        self.bytes.resize(off + length, 0)
        memcpy(dest=self.bytes.unsafe_ptr() + off, src=ptr, count=length)
        self.spans.append(TokenSpan(off, length))
        return idx
