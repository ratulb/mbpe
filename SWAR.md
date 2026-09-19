# SWAR, or: how `_hasless` and `_haszero` actually work

There's a function in `bpe/pretokenizer.mojo` called `_hasless`. It takes two `UInt64`s and returns a `UInt64`. It's four lines long:

```mojo
@always_inline
def _hasless(x: UInt64, n: UInt64) -> UInt64:
    return (x - n * UInt64(0x0101010101010101)) & ~x & UInt64(0x8080808080808080)

```

Its sibling `_haszero` is the same shape, with `n` fixed at 1:

```mojo
@always_inline
def _haszero(x: UInt64) -> UInt64:
    return (x - UInt64(0x0101010101010101)) & ~x & UInt64(0x8080808080808080)
```

These two functions are the arithmetic core of mbpe's letter-run scanner. They let the tokenizer classify eight bytes at once instead of one at a time, and they're the reason the encode path is several times faster than a per-byte loop would be. But their purpose isn't obvious from the source, and their correctness isn't obvious either. This write-up unpacks them.

We'll start with the problem they solve, build up to the technique, walk through the arithmetic bit by bit on a concrete example, and then — because Mojo has a first-class SIMD type — compare the approach they use (SWAR) with the vector approach you might reach for instead.

## The problem: too many bytes, too few comparisons

The pre-tokenizer's job, at its core, is to look at each byte of the input and decide what class it belongs to. Is it a letter? A digit? Whitespace? A punctuation mark? Based on those decisions, it finds runs of like bytes and emits them as units.

The naive way to do this is a loop:
```mojo
while i < n:
    var b = input[i]
    if is_letter(b):
        i += 1
    else:
        break

```
That's one byte per iteration. If you're scanning a 5 MB text and most of it is letters, you're doing millions of iterations, each one paying for a load, a comparison, and a branch. The branch is the expensive part — it's unpredictable, and the CPU's branch predictor can't help if the input is mixed.

The fix is to look at more than one byte at a time. Modern CPUs are 64-bit; a single register can hold eight bytes if you're willing to treat them as eight lanes. If you could test all eight lanes in parallel, you'd classify eight bytes in the time it currently takes to classify one.

That's the premise of SWAR: **SIMD Within A Register**. You don't need special vector instructions to process multiple values in parallel — you just need arithmetic that doesn't let the lanes interfere with each other. And that's exactly what `_hasless` and `_haszero` provide.

## The setup: eight bytes in one register
Take a `UInt64`. It's 64 bits. Number the bits 0 to 63, with bit 63 being the most significant. Now split it into eight groups of eight bits:

```
 bit 63      bit 56 bit 55      bit 48  ...  bit 7       bit 0
 ┌─────────────┬─────────────┬───┬─────────────┬─────────────┐
 │  byte 7     │  byte 6     │ … │  byte 1     │  byte 0     │
 └─────────────┴─────────────┴───┴─────────────┴─────────────┘
 ```

 Each group is a byte. If you load eight consecutive bytes of text into this register (with a bit of `unsafe` pointer casting that `mbpe` does), you now have eight characters in one variable.

The question is: how do you ask a question about all eight of them at once?

You can't just do if `x < some_value` — that would compare the 64-bit integer as a whole. What you want is a comparison that's per-byte, that returns a per-byte answer. Ideally, it returns a mask where bit 7 (the top bit) of byte `i` is set if byte `i` satisfies the condition, and clear otherwise. Then you can test the whole mask with a single `!= 0`, or find the first matching byte with a `trailing_zeros` call.

That's exactly what `_haszero` and `_hasless` do.

## The technique: borrow propagation as a feature
Here's the key insight. When you subtract 1 from a byte:

If the byte is 0, `0 - 1` underflows and becomes `0xFF`. The top bit flips from 0 to 1.

If the byte is 1 or more, `b - 1` stays in range and the top bit doesn't flip.

That's the arithmetic foundation. If you could subtract 1 from every byte independently, you'd have a per-byte test for "is this byte zero?" — just check if the top bit flipped.

But subtraction borrows across byte boundaries. If byte 0 is 0 and you subtract 1 from the whole register, byte 0 underflows (becomes 0xFF) and borrows 1 from byte 1. Then byte 1's effective value is (byte 1 - 1), and if byte 1 was 0, it underflows too, and borrows from byte 2, and so on.

The naive reaction is: borrow propagation is a problem, it corrupts the lanes. The SWAR trick is: `borrow propagation only happens when a lane is zero`. And a lane that's zero is already a lane we want to flag. So the borrow isn't corruption — it's a signal that coincides with the answer.

Let me make that concrete.

## `_haszero`, step by step
Take a register with four bytes (I'll use four instead of eight for readability; the arithmetic is identical for any number of lanes).

Let `x = 0x41004243`. In bytes:

```
byte 3 = 0x41   ('A')
byte 2 = 0x00   (NUL)
byte 1 = 0x42   ('B')
byte 0 = 0x43   ('C')
```

Only byte 2 is zero. We want the function to return a mask where the top bit of byte 2 is set.

### Step 1: subtract `0x01010101`

```mojo
x - 0x01010101
```

`0x01010101` is a register where every byte is 1. Subtracting it is like subtracting 1 from every byte — except for the borrow issue. Let's compute it manually.

```
  0x41 00 42 43
- 0x01 01 01 01
= ?
```

Do it byte by byte from the least significant end, allowing borrows:

- **Byte 0**: `0x43 - 0x01 = 0x42`. No borrow.

- **Byte 1**: `0x42 - 0x01 = 0x41`. No borrow.

- **Byte 2**: `0x00 - 0x01 = 0xFF` with a borrow. The borrow propagates to byte 3.

- **Byte 3**: `0x41 - 0x01 - 0x01 = 0x3F`. The second - `0x01` is the incoming borrow.

Result: `0x3F FF 41 42`.

In bytes:

```
byte 3 = 0x3F
byte 2 = 0xFF   ← this byte was 0, now has top bit set
byte 1 = 0x41
byte 0 = 0x42

```

Notice: byte 2's top bit is now set, because the underflow made it `0xFF`. Byte 3 also changed (`0x41 → 0x3F`), because it absorbed the borrow. But byte 3's top bit is 0 — 0x3F has its high bit clear.

### Step 2: AND with `~x`

```
(x - 0x01010101) & ~x

```
`~x` is the bitwise NOT of `x`. For our bytes:

```
x    = 0x41 00 42 43
~x   = 0xBE FF BD BC

```

Now AND:

```
  0x3F FF 41 42
& 0xBE FF BD BC
= 0x3E FF 01 00

```
In bytes:

```
byte 3 = 0x3E
byte 2 = 0xFF   ← still has top bit set
byte 1 = 0x01
byte 0 = 0x00

```

For byte `i`:

- If byte `i` of x was 0: `~x` has all bits set in that lane, so (`x - 1`) passes through unchanged. If (`x - 1`) had its top bit set (because the byte was 0), the top bit survives the AND.

- If byte `i` of `x` was nonzero: `~x` has some bits clear. This tends to clear bits in (`x - 1`), including the top bit if the byte absorbed a borrow.

Byte 3 was `0x41`, absorbed a borrow (becoming `0x3F`), and the AND with `~0x41 = 0xBE` turned it into `0x3E`. Top bit of `0x3E` is 0, so byte 3 isn't flagged. Byte 2 was 0, became `0xFF`, and `~x` for that lane is `0xFF`, so it passes through unchanged. Top bit set.

### Step 3: mask the top bits

```
(...) & 0x80808080
```

`0x80` in binary is `10000000`. `0x80808080` is a register where every byte is `0x80`, i.e. only the top bit of each lane is set.


```
  0x3E FF 01 00
& 0x80 80 80 80
= 0x00 80 00 00

```
Result: `0x00800000`. Byte 2 has its top bit set. Bytes 0, 1, and 3 have theirs clear. The mask says: byte 2 is zero, and that's the only one.

If you wanted to find which byte matched, you'd call `trailing_zeros` on this mask, get 23, and divide by 8 to get byte index 2. If you just want to know whether any byte matched, you'd check `mask != 0`.

That's `_haszero`.

## `_hasless`, generalized
`_haszero(x)` is really `_hasless(x, 1)` — "which bytes are less than 1?", i.e. "which bytes are zero?".

`_hasless(x, n)` generalizes to "which bytes are less than `n`?" for any small `n`. The only change is the subtraction operand:

```
x - n * 0x01010101

```
n * 0x01010101 broadcasts the byte value n into every lane. For n = 5, it's 0x05050505. Subtracting this from x subtracts 5 from every byte, with the same borrow semantics as before: a byte less than 5 underflows and gets its top bit set.

The rest of the function is identical: & ~x filters borrow contamination, & 0x80808080 isolates the top bits.

Two caveats:

1. `n` must be less than 128. The whole trick assumes "underflow sets the top bit." If `n` itself has its top bit set, the arithmetic breaks. For character classification, `n` is always a character code — `0x7B` for `'{'`, `0x61` for `'a'` — and those are well under 128. So the constraint is satisfied by construction.
2. Bytes must be treated as unsigned. If you interpret the lanes as signed, "the top bit set" means "negative," and the comparison goes sideways. In mbpe the inputs are loaded via `unsafe_bitcast[UInt64]`, which is explicitly unsigned.

## How mbpe uses it

The function in `bpe/pretokenizer.mojo` that puts `_hasless` and `_haszero` to work is `_letters8`:

```mojo
@always_inline
def _letters8(x: UInt64) -> UInt64:
    var l = x | UInt64(0x2020202020202020)
    var below_z = _hasless(l, UInt64(0x7B))
    var is_0x7B = _haszero(l ^ (UInt64(0x7B) * UInt64(0x0101010101010101)))
    below_z = below_z & ~is_0x7B
    var below_a = _hasless(l, UInt64(0x61))
    return below_z & ~below_a
```

This function takes eight bytes and returns a mask where the top bit of byte i is set if byte i is an ASCII letter (a–z or A–Z). It classifies eight bytes in about six operations.

Walk through it:

1. `l = x | 0x2020...` — OR every byte with `0x20`. This is a case-folding trick: ASCII uppercase letters (`0x41`–`0x5A`) differ from their lowercase counterparts (`0x61`–`0x7A`) exactly by bit 5, which is `0x20`. After this step, all letters are in the range `0x61`–`0x7A`.
2. `below_z = _hasless(l, 0x7B)` — flag every byte less than `0x7B` (`'{'`). This includes `a`–`z`, but also every byte below `'a'` — digits, punctuation, control characters.
3. `is_0x7B = _haszero(l ^ 0x7B...)` — the exact-equality mask for `'{'`. This is a correction: `_hasless` has a false positive for a byte exactly equal to the bound, when a lower byte borrows into it. The `'{'` case is common, so it's worth fixing. XOR-ing with `0x7B` in every lane produces 0 for bytes that were `'{'`, and the `_haszero` then flags those.
4. `below_z &= ~is_0x7B` — remove `'{'` from the "less than `'z'`" set. Now `below_z` means "byte ≤ `'z'`".
5. `below_a = _hasless(l, 0x61)` — flag every byte less than `'a'`.
6. `return below_z & ~below_a` — intersect. "≤ `'z'`" AND "≥ `'a'`" = "`'a'` ≤ byte ≤ `'z'`" = an ASCII letter.

That's eight bytes classified in six operations. Compare with the per-byte loop: eight iterations, eight loads, eight comparisons, eight branches. For long runs of ASCII letters, `_letters8` is roughly an order of magnitude faster.

## Where it goes from there
`_letters8` returns a mask. The pre-tokenizer's `_swar_letter_run` uses that mask in a loop:

```mojo
while j + 8 <= n:
    var w: UInt64 = (p8.unsafe_offset(j)).unsafe_bitcast[UInt64]()[]
    var nl = ~_letters8(w) & UInt64(0x8080808080808080)
    if nl != 0:
        return consumed + (pop_count(lsb - 1) >> 3)
    consumed += 8
    j += 8
```

The pattern: load 8 bytes, classify them all, check if any of them is not a letter (`nl != 0`). If all 8 are letters, skip ahead 8 bytes and repeat. If some byte in the group isn't a letter, `trailing_zeros(nl) >> 3` gives the index of the first non-letter byte within the group — and the run ends there.

This is the hot path of the encoder for letter runs. On text that's mostly letters, it scans eight bytes per iteration.

## Now: why not use SIMD?
Mojo has a first-class SIMD type. It's not a toy — it maps directly to hardware vector registers, it's type-safe and zero-cost, and it works across CPU architectures without manual intrinsics. So the natural question is: why does mbpe use SWAR instead of `SIMD[UInt8, 8]`?

The answer is that for this specific problem — 8-byte scans with a variable stopping condition — the two approaches are close enough that the choice is mostly stylistic, and SWAR wins on a few small margins. But it's not an obvious win, and at larger widths the answer flips. Here's the side-by-side.

## The same problem, two ways
Let's define the problem concretely: scan 8 bytes at a time, classify each as "letter or not," and find the first non-letter.

**The SWAR version** (what mbpe does):

```mojo
var w: UInt64 = p8.unsafe_offset(j).unsafe_bitcast[UInt64]()[]
var nl = ~_letters8(w) & 0x8080808080808080
if nl != 0:
    return consumed + (pop_count(lsb - 1) >> 3)

```

Six arithmetic ops for the classification, plus a bit-clear, plus a branch, plus (when the branch fires) a `trailing_zeros`-style extraction to find the lane index.

The **SIMD version** (what you might write first):

```mojo
var w = p8.unsafe_offset(j).unsafe_bitcast[SIMD[UInt8, 8]]()[]
var is_letter = _letters8_simd(w)  # returns SIMD[Bool, 8]
if not is_letter.reduce_and():
    # find the first false lane
    var mask = is_letter.to_bits()    # SIMD[UInt64, 1] or scalar
    var idx = trailing_zeros(~mask) >> 3
    return consumed + idx
```

The classification itself is a couple of vector ops, and `reduce_and` collapses the mask to a single boolean. But to find which lane failed, you have to pull the mask back out of the vector unit and into a scalar, then do a `trailing_zeros` — which is exactly what the SWAR version did, but with an extra hop through the vector register.

That extra hop is the crux of the comparison.

### Side-by-side

| | SWAR | SIMD |
|---|---|---|
| What it operates on | General-purpose 64-bit integer registers | Vector registers (128/256/512-bit, depending on CPU) |
| Instruction width | Fixed at 64 bits (8 bytes) | Configurable (8, 16, 32, 64 bytes depending on type) |
| Classification cost | ~6 scalar ops for 8 bytes | ~2 vector ops for 8–64 bytes |
| Extracting "which lane failed" | Free — the mask is already a `UInt64`; `trailing_zeros` directly | Requires moving the mask out of the vector register (`to_bits`), then `trailing_zeros` |
| Portability | Universal — every 64-bit CPU has the needed ops | Portable at the API level; performance varies by target |
| Register pressure | Uses scalar integer registers, which are often idle | Uses vector registers, which may be contended |
| Code clarity | Cryptic — the borrow trick is non-obvious | More readable — `w.lt(0x61)` says what it means |
| Compile-time const | `n` must be < 128; mask must be unsigned | No such constraint; comparison is signed or unsigned per type |
| Scales to larger widths | No — 64 bits is the limit of a scalar register | Yes — `SIMD[UInt8, 32]` is a single 256-bit operation on AVX2 |
| Fallback when unavailable | Never needed — 64-bit integer arithmetic always works | Compiler splits large SIMD across available registers, which can be slow |

### Pros of SWAR for this problem

- **No boundary cost.** The mask is already a `UInt64`, so `trailing_zeros`, `pop_count`, and the bit-masking that follows are all in the same arithmetic domain. No conversions.
- **Uses idle resources.** The scalar integer units are separate from the vector units. If the surrounding code is doing vector work elsewhere, SWAR runs in parallel rather than contending.
- **Uniform performance.** Every machine Mojo supports has the same 64-bit integer arithmetic. No target-specific surprises.
- **Small code.** The whole primitive is a handful of lines, inlined into the caller.

### Cons of SWAR for this problem

- **Cryptic.** The borrow-propagation trick is not obvious. A reader has to derive it or be told.
- **Constraints.** `n` < 128 and unsigned lanes are required. Both are fine here but not in general.
- **Fixed at 8 bytes.** No headroom. If you wanted 16 or 32 bytes per scan, SWAR can't help — you'd chain two or four 64-bit values, which is more code and no more efficient than a single wider SIMD op.
- **Non-obvious correctness.** The `& ~x` cleanup step is doing real work, and it's not obvious without careful reasoning.

### Pros of SIMD for this problem

- **Readable.** `w.lt(n)` says what it does. The mask is a `SIMD[Bool, 8]`, and `reduce_and()` gives a clean boolean answer.
- **Scales.** A `SIMD[UInt8, 32]` classifies 32 bytes in the same number of operations as 8. If the scan were longer per iteration, SIMD would win big.
- **No arithmetic constraints.** Signed comparisons work. Bounds are whatever the type allows.
- **Composable.** Shuffles, reductions, selects, and interleaves are all available if you need to do more complex mask manipulation.

### Cons of SIMD for this problem

- **Boundary cost.** To find "which lane failed," you have to move the mask out of the vector unit and into a scalar. That's a `to_bits` or `as_bytes`, then a `trailing_zeros`. The SWAR version avoids this because the mask was born scalar.
- **Register pressure.** If the surrounding code uses vector registers, the SIMD version contends with it. SWAR uses a different set of registers.
- **Target variance.** A `SIMD[UInt8, 8]` on a machine with 128-bit vector registers wastes half the register. On a machine with 64-bit vector registers (some ARM cores), it fits exactly. On a machine without SIMD support, the compiler emits scalar fallbacks — which, in the worst case, are slower than the SWAR version because they don't exploit the trick.
- **Larger code size.** The SIMD version has more types involved (`SIMD[UInt8, 8]`, `SIMD[Bool, 8]`, conversions between them), and more operations at the boundaries. Not a lot, but more than the SWAR version.

### The honest verdict
For an 8-byte scan, the two are close. The SWAR version wins slightly because the mask never leaves the scalar domain, and because scalar registers are typically less contended. The SIMD version wins slightly on readability and would win more if the scan were wider.

For a 16-byte or 32-byte scan, SIMD wins. The per-operation cost is roughly the same, but the SIMD version processes two to four times as many bytes per operation. The boundary cost gets amortized, and the SWAR version can't keep up — you can't fit 16 bytes in a scalar register.

The choice depends on the width. mbpe uses SWAR because its scan is 8 bytes and its primitive is short. If the same code were written for a wider scan (say, to speed up long letter runs further), the natural choice would be `SIMD[UInt8, 32]` or larger, and the arithmetic would change shape.

This is worth stating plainly because it's easy to fall into "one technique is better" reasoning. Neither is universally better. They're optimized for different regimes, and the right choice depends on the size of the data unit you're processing.

## What this means for a novice

If you're reading this and thinking "which one should I use?", here's a simple heuristic:

- If your scan is 8 bytes or less, and you can express the test as arithmetic on `UInt64`, SWAR is fine. The code is short, it works everywhere, and the boundary cost of SIMD isn't worth paying.
- If your scan is 16 bytes or more, or you need signed comparisons or multi-lane logic, use SIMD. The wider width pays for the boundary cost, and the readable syntax makes the code easier to maintain.
- If you're not sure, start with SIMD. It's the more modern tool, it's more readable, and the performance gap at small widths is small enough that you won't notice. SWAR is a micro-optimization you reach for when you've measured the alternatives and decided the extra opacity is worth the marginal gain.

mbpe chose SWAR because it's a small fixed-width scan in a hot loop, and the primitive is short enough to document. That's the honest reason, and it's a reason that depends on the specific numbers. Your numbers will be different.

## The general lesson
SWAR is a technique with a long history outside this project. The same borrow-propagation trick appears in cryptographic implementations, in string search (the memchr function in many C libraries uses a variant), and in bit-manipulation libraries. The pattern is worth internalizing: when you want to compare N values in parallel with scalar arithmetic, look for a way to make the lanes interfere with each other in a controlled, detectable way.

But internalize it as one tool, not the tool. SIMD does the same job with different tradeoffs. The best way to think about it is: both techniques let you process multiple values at once. SWAR does it using the scalar integer registers you already have; SIMD does it using vector registers you might have. The first works everywhere and is limited to 64 bits; the second is faster at wider widths but has boundary costs and target variance.

In `_haszero` and `_hasless`, the interference is borrow propagation, and the detection is "did the top bit flip?" The `& ~x` step is the cleanup that makes the answer reliable despite the interference. It's four lines of arithmetic and one bit of insight: subtraction underflows exactly when a byte is small, and the underflow is visible in the top bit.

The next time you find yourself with a loop that processes bytes one at a time, consider both. For a quick 8-byte scan, `_hasless` might be what you want. For anything wider, reach for `SIMD[UInt8, N]` and let the compiler do the work.
