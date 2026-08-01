"""Minimal growable typed array with zero-copy views.

Vendored and genericized from tenmo's `IntArray`
(`intarray.mojo`), trimmed to the operations the tokenizer needs:

- growable storage with 1.5x amortized growth (`append`/`reserve`)
- `_capacity == -1` marks a non-owning VIEW (borrowed pointer,
  zero-alloc copy, no-op destroy)
- `__getitem__(Slice)` returns a zero-copy view for contiguous
  forward slices (deep copy for strided/negative steps)
- bounds-checked access with negative indexing
- `from_list`/`tolist` convert to/from stdlib `List` (client API)
- element storage is always on the heap; the struct itself is a
  plain (non-register-passable) copyable value, like stdlib `List`
- iterable: `for x in arr` yields `ref[origin] Self.T` elements; the
  iterator follows the b2 stdlib `_ListIter` pattern (type param
  after `//`, `downcast` in the `IteratorType` member)

Aliases: `IntArray = Array[Int]`, `ByteArray = Array[Byte]`.
"""

from std.os import abort
from std.memory import alloc, memcpy, memmove
from std.builtin.rebind import downcast


def _panic(*msgs: String):
    for m in msgs:
        print(m)
    abort("")


struct Array[T: ImplicitlyCopyable & Intable & ImplicitlyDeletable](
    ImplicitlyCopyable, Iterable, Sized, Writable
):
    """A lightweight, growable array of `Self.T` backed by a heap buffer."""

    var _data: Optional[UnsafePointer[Self.T, MutAnyOrigin]]
    var _size: Int  # Current number of elements
    var _capacity: Int  # Allocated capacity

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = ArrayIter[
        downcast[Self.T, Copyable], iterable_origin, True
    ]

    @always_inline("nodebug")
    def __init__(out self, size: Int = 0):
        """Create an array with `size` uninitialized elements (0 = empty)."""
        if size > 0:
            self._data = alloc[Self.T](size)
        else:
            self._data = {}
        self._size = size
        self._capacity = size

    @always_inline("nodebug")
    def __init__(out self, *values: Self.T):
        """Create an array from variadic arguments: Array(1, 2, 3)."""
        var n = len(values)
        self._data = alloc[Self.T](n)
        self._size = n
        self._capacity = n
        var data = self._data.unsafe_value()
        for i in range(n):
            data[i] = values[i]

    @always_inline("nodebug")
    def __init__(out self, values: List[Self.T]):
        """Create an owning array from a stdlib List (memcpy, O(n))."""
        var n = len(values)
        self._data = alloc[Self.T](n)
        self._size = n
        self._capacity = n
        if n > 0:
            memcpy(
                dest=self._data.unsafe_value(), src=values.unsafe_ptr(), count=n
            )

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """Copy constructor.

        Owning arrays get a deep copy (alloc + memcpy).
        View arrays get a shallow copy — just the pointer, zero alloc.
        """
        self._size = copy._size
        self._capacity = copy._capacity
        if copy.owning():
            self._data = alloc[Self.T](copy._capacity)
            if copy._size > 0:
                memcpy(
                    dest=self._data.unsafe_value(),
                    src=copy._data.unsafe_value(),
                    count=copy._size,
                )
        else:
            self._data = copy._data

    @always_inline("nodebug")
    def __del__(deinit self):
        """Free memory only if owning. Views are no-op."""
        if self.owning() and self._data:
            self._data.unsafe_value().free()

    @staticmethod
    @always_inline("nodebug")
    def from_list(values: List[Self.T]) -> Self:
        """Build an owning array from a stdlib List (memcpy, O(n))."""
        var result = Self(len(values))
        if len(values) > 0:
            memcpy(
                dest=result._data.unsafe_value(),
                src=values.unsafe_ptr(),
                count=len(values),
            )
        return result^

    @staticmethod
    @always_inline("nodebug")
    def with_capacity(capacity: Int) -> Self:
        """Pre-allocated capacity, zero length (no realloc on append)."""
        var result = Self()
        if capacity > 0:
            result._data = alloc[Self.T](capacity)
            result._capacity = capacity
        return result^

    @always_inline("nodebug")
    def size(self) -> Int:
        """Number of elements."""
        return self._size

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Number of elements (enables len())."""
        return self._size

    @always_inline("nodebug")
    def capacity(self) -> Int:
        """Allocated capacity; 0 for views."""
        return self._capacity if self._capacity >= 0 else 0

    @always_inline("nodebug")
    def is_empty(self) -> Bool:
        return self._size == 0

    @always_inline("nodebug")
    def owning(self) -> Bool:
        """`_capacity >= 0` = owning (empty or allocated).

        `_capacity == -1` = view (borrowed pointer, zero-alloc copy,
        no-op destroy). Empty owning arrays (`_size=0, _capacity=0`)
        can still grow.
        """
        return self._capacity >= 0

    @always_inline("nodebug")
    def owned_copy(self) -> Self:
        """Return an owning copy (deep copy if view, no-op if owning)."""
        if self.owning():
            return self
        var result = Self.with_capacity(self._size)
        if self._size > 0:
            memcpy(
                dest=result._data.unsafe_value(),
                src=self._data.unsafe_value(),
                count=self._size,
            )
            result._size = self._size
        return result^

    @always_inline("nodebug")
    def unsafe_ptr(self) -> UnsafePointer[Self.T, MutAnyOrigin]:
        """Raw pointer to the data (dangling for empty arrays)."""
        if self._data:
            return self._data.unsafe_value()
        return UnsafePointer[Self.T, MutAnyOrigin].unsafe_dangling()

    @always_inline("nodebug")
    def __getitem__(ref self, idx: Int) -> ref[self] Self.T:
        """Element at index (negative indexing; bounds-checked)."""
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            _panic("Array: index out of bounds")
        return (self._data.unsafe_value() + index)[]

    @always_inline("nodebug")
    def __setitem__(mut self, idx: Int, value: Self.T):
        """Set element at index (negative indexing; bounds-checked).

        Raises:
            Panic if index is out of bounds or array is a view.
        """
        if not self.owning():
            _panic("Array: can't modify a view")
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            _panic("Array: index out of bounds")
        (self._data.unsafe_value() + index)[] = value

    @always_inline("nodebug")
    def __getitem__(ref self, slice: Slice) -> Self:
        """Slice into a view (step==1) or deep copy (step != 1).

        Contiguous slices return a view — zero-alloc, just a borrowed pointer.
        Strided slices deep-copy. Supports negative indices.
        """
        var step = slice.step.or_else(1)

        if step == 0:
            _panic("Array: slice step cannot be zero")

        var start: Int
        var stop: Int

        if step > 0:
            start = slice.start.or_else(0)
            stop = slice.end.or_else(self._size)
        else:
            start = slice.start.or_else(self._size - 1)
            stop = slice.end.or_else(-self._size - 1)

        if start < 0:
            start += self._size
        if stop < 0:
            stop += self._size

        var size: Int
        if step > 0:
            size = max(0, (stop - start + step - 1) // step)
        else:
            size = max(0, (start - stop - step - 1) // (-step))

        # Contiguous forward slice → return a view
        if step == 1:
            if size == 0:
                return Self()
            var result = Self()
            if self._data:
                result._data = self._data.unsafe_value() + start
            result._size = size
            result._capacity = -1  # marks as view
            return result^

        # Strided or negative step → deep copy
        var result = Self.with_capacity(size)
        var src_idx = start
        var data = self._data.unsafe_value()

        if step > 0:
            while src_idx < stop and src_idx < self._size:
                if src_idx >= 0:
                    result.append(data[src_idx])
                src_idx += step
        else:
            while src_idx > stop and src_idx >= 0:
                if src_idx < self._size:
                    result.append(data[src_idx])
                src_idx += step

        return result^

    @always_inline("nodebug")
    def reserve(mut self, required: Int):
        """Ensure capacity, reallocating if needed.

        Raises:
            Panic if called on a non-owning (view) array.

        Note:
            Growth strategy: new capacity = max(required, current_capacity * 1.5 + 1)
        """
        if not self.owning():
            _panic("Array: can't reserve on a view")
        if required <= self._capacity:
            return

        var new_cap = max(required, self._capacity * 3 // 2 + 1)
        var new_data = alloc[Self.T](new_cap)

        if self._size > 0:
            memcpy(
                dest=new_data, src=self._data.unsafe_value(), count=self._size
            )

        if self._data:
            self._data.unsafe_value().free()
            self._data = {}
        self._data = new_data
        self._capacity = new_cap

    @always_inline
    def append(mut self, *values: Self.T):
        """Append one or more elements (amortized 1.5x growth)."""
        self.reserve(self._size + len(values))
        var data = self._data.unsafe_value()
        for i in range(len(values)):
            data[self._size + i] = values[i]
        self._size += len(values)

    @always_inline("nodebug")
    def clear(mut self):
        """Reset size to zero (capacity retained); panics on views."""
        if not self.owning():
            _panic("Array: can't clear a view")
        self._size = 0

    @always_inline("nodebug")
    def tolist(self) -> List[Self.T]:
        """Convert to a stdlib List (copy, O(n))."""
        var result = List[Self.T](capacity=self._size)
        var data = self._data.unsafe_value()
        for i in range(self._size):
            result.append(data[i])
        return result^

    @always_inline("nodebug")
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the elements (yields `ref[origin] Self.T`)."""
        return ArrayIter(
            index=0,
            src=rebind[UnsafePointer[Self.T, origin_of(self)]](
                self.unsafe_ptr()
            ),
            length=self._size,
        )

    @no_inline
    def __str__(self) -> String:
        if self._size == 0:
            return "[]"
        var result = String("[")
        var data = self._data.unsafe_value()
        for i in range(self._size):
            if i > 0:
                result += ", "
            result += String(Int(data[i]))
        result += "]"
        return result

    @no_inline
    def __repr__(self) -> String:
        return self.__str__()

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())


@fieldwise_init
struct ArrayIter[
    mut: Bool,
    //,
    T: Copyable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator for Array (b2 stdlib `_ListIter` pattern).

    Parameters:
        mut: Whether the reference to the array is mutable.
        Self.T: The type of the elements in the array.
        origin: The origin of the array.
        forward: The iteration direction. `False` is backwards.
    """

    comptime Element = Self.T

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: UnsafePointer[Self.T, Self.origin]
    var length: Int

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        comptime if Self.forward:
            if self.index >= self.length:
                raise StopIteration()
            self.index += 1
            return self.src[self.index - 1]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            return self.src[self.index]

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        comptime if Self.forward:
            iter_len = self.length - self.index
        else:
            iter_len = self.index

        return (iter_len, {iter_len})

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        comptime if Self.forward:
            return self.length - self.index
        else:
            return self.index


comptime IntArray = Array[Int]
comptime ByteArray = Array[Byte]
