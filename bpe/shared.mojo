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

    var bytes: ByteArray
    var spans: List[TokenSpan]

    def __init__(out self):
        self.bytes = ByteArray()
        self.spans = List[TokenSpan]()

    def __init__(out self, *, copy: Self):

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

        var idx = len(self.spans)
        var off = len(self.bytes)
        self.bytes.resize(off + length, 0)
        memcpy(dest=self.bytes.unsafe_ptr() + off, src=ptr, count=length)
        self.spans.append(TokenSpan(off, length))
        return idx

