# UNICODE_EVENT_POINTS.md — How the event-point table is built

What problem the bpe/unicode_tables.mojo solves, why the naive
representation is wasteful, what the "event point" idea is, and then a
worked example at real scale. Code references are to
`scripts/gen_unicode_tables.py` (`build_table`) and `bpe/unicode_tables.mojo`
(`_class_mask`).

## 1. The problem, in plain English

The pre-tokenizer needs to answer questions about characters. Is this
character a letter? A digit? Whitespace? A combining mark? Those questions
come up constantly during encode and train, and the answers have to be
exactly right for every character in Unicode — not just the ones you
thought to test.

Unicode defines around 1.1 million codepoints. The classification questions
the tokenizer needs to ask are defined by *properties* (letter, digit,
lowercase, uppercase, mark, whitespace) that the Unicode standard specifies
per codepoint. A straightforward way to answer "is codepoint X a letter?"
would be to look up X in a table that has one entry per codepoint.

That's the naive design: a 1.1-million-entry array, where each entry is a
small integer saying which classes that codepoint belongs to. It works. It's
also wasteful — the array is roughly a megabyte, and the vast majority of it
is repetition.

Why repetition? Because Unicode properties change slowly across the
codepoint space. The letters `A` through `Z` are all letters; the letters
`a` through `z` are all letters, and also lowercase; the digits `0` through
`9` are all digits. Long runs of codepoints share the same class membership.
A codepoint's membership only *changes* at a few thousand places in the
entire space, not a million.

The event-point table is what you get when you notice this and exploit it:
instead of storing a mask per codepoint, store a mask per *run* of
codepoints that share the same mask. Each run is defined by where it
starts, and the run continues until the mask changes again. The collection
of "places where something changes" are the **event points**, and the table
is the sorted list of them plus the mask each one introduces.

That's the whole idea. The rest of this file explains how the event points
are collected, how the runs are built, and how lookup works, with a small
worked example.

## 2. Definitions

A few terms, defined once so the worked example can use them without
re-explaining.

- **Codepoint.** A Unicode scalar value, from `U+0000` to `U+10FFFF`. The
  full range is about 1.1 million values.
- **Class.** One of six Unicode properties the tokenizer cares about.
  `L` = letter, `N` = number, `l` = lowercase letter, `u` = uppercase
  (or titlecase) letter, `M` = combining mark, `W` = whitespace. Each
  class is assigned one bit in a 6-bit mask.
- **Class interval.** A contiguous range of codepoints `(lo, hi)` — both
  inclusive — that all belong to one class. For example, `L` includes the
  interval `[0x41, 0x5A]`, which is `A` through `Z`.
- **Event point (or cut point).** A codepoint where at least one class's
  membership changes. Since intervals are inclusive, membership turns
  *on* at `lo` and *off* just after `hi` — so each interval contributes
  two event points: `lo` and `hi + 1`. The `+1` converts an inclusive
  interval `[lo, hi]` into a half-open segment `[lo, hi+1)`, which tile
  the codepoint space with no gaps and no overlaps.
- **Segment.** The half-open range between two consecutive event points,
  `[p[i], p[i+1])`. By construction, no class boundary falls inside a
  segment, so every codepoint in it has the same mask.
- **Coalescing.** If two adjacent segments happen to have the same mask,
  they're merged into one table entry. Only *changes* in the mask produce
  new entries.

## 3. Why event points, and not something simpler

Two alternatives are worth naming so the design choice is clear.

**Alternative A: one entry per codepoint.** A 1.1-million-entry array,
each entry a byte. ~1.1 MB of storage. Lookup is a direct index — fast —
but the table is large, and it's almost entirely repetitive. For a
tokenizer that ships with three vocabularies already taking tens of
megabytes on disk, an extra megabyte isn't fatal, but it's not necessary
either.

**Alternative B: one entry per class interval.** Store each class's
intervals separately, and answer "is X in class C?" by binary-searching
class C's intervals. That's what the *original* hand-written if-chains
did, one chain per class. It's compact per class, but a single query
requires up to six binary searches — one per class — and the classes
aren't sorted the same way, so the searches don't share work.

**The event-point table is a third option.** One lookup answers all six
questions at once, because each entry encodes the full mask. Storage is
tiny (a few thousand entries), lookup is a single binary search, and the
structure is the natural consequence of noticing that masks change slowly.

The event-point construction is what makes both properties possible:
compact storage because long runs collapse to one entry, and single-lookup
classification because the mask at any point answers every class question
simultaneously.

## 4. How the table is built

Three steps, applied to the collection of all class intervals across all
six classes.

**Step 1 — Collect event points.** Every interval `(lo, hi)` of every class
contributes two points: `lo` and `hi + 1`. The union of all these points,
plus `0` (so codepoints below the first real boundary resolve to mask
zero), is the set of event points for the whole table.

**Step 2 — Sweep and coalesce.** Walk the event points in sorted order.
For each segment between consecutive points, compute the mask by asking
every class "does this segment's range intersect your intervals?" If the
mask is the same as the previous segment's, merge the two — the segment
doesn't start a new run. If it's different, emit a new `BOUNDS` entry with
the segment's start and a new `MASKS` entry with the mask.

**Step 3 — Emit.** The result is two parallel arrays: `BOUNDS` (sorted
starts of the constant-mask runs) and `MASKS` (the mask for each run).
Lookup is a binary search over `BOUNDS` to find the largest entry not
exceeding the query codepoint, then read the corresponding `MASKS` entry.

The next section walks this end to end on a small window.

## 5. Worked example: U+0040–U+0062, three classes

To keep the arithmetic readable, restrict attention to three of the six
classes and a small window: `[0x40, 0x63)`. (The other three classes —
`N`, `M`, `W` — have no members in this window, so they contribute no
event points here and their bits stay 0 throughout.)

Class intervals intersecting the window:

| Class | Intervals in window |
|---|---|
| `L` (`\p{L}`, bit `0x01`) | `[0x41, 0x5A]`, `[0x61, 0x7A]` |
| `l` (`\p{Ll}`, bit `0x04`) | `[0x61, 0x7A]` |
| `u` (`\p{Lu}\|\p{Lt}`, bit `0x08`) | `[0x41, 0x5A]` |

**Step 1 — collect event points.** Each interval emits `lo` and `hi + 1`:

- `L`: `0x41`, `0x5B`, `0x61`, `0x7B`
- `l`: `0x61`, `0x7B`
- `u`: `0x41`, `0x5B`

Union, clipped to the window (with the window start `0x40` added so the
first segment has a left edge): `{0x40, 0x41, 0x5B, 0x61, 0x63}`.

**Step 2 — sweep.** Evaluate the mask at each event point. `mask_at` in
`build_table` does this by binary-searching each class's interval starts
and OR-ing in the class's bit if the point falls inside one of its
intervals:

| Segment | Members | `L` | `l` | `u` | Mask |
|---|---|---|---|---|---|
| `[0x40, 0x41)` | `@` | 0 | 0 | 0 | `0x00` |
| `[0x41, 0x5B)` | `A`–`Z` | 1 | 0 | 1 | `0x01\|0x08 = 0x09` |
| `[0x5B, 0x61)` | `[\]^_\`` | 0 | 0 | 0 | `0x00` |
| `[0x61, 0x63)` | `a`, `b` | 1 | 1 | 0 | `0x01\|0x04 = 0x05` |

**Step 3 — coalesce.** In this window no two adjacent segments share a
mask (`0x00`, `0x09`, `0x00`, `0x05` alternate), so all four survive as
separate entries. In the full table this is where the compression happens:
thousands of fine segments collapse wherever neighbours happen to agree
(e.g. the long CJK letter runs, where dozens of consecutive intervals
produce the same mask).

Note that the two `0x00` segments are *not* adjacent — the `0x09` segment
sits between them — so they can't be coalesced into one entry even though
they have the same mask.

**Step 4 — emit.** The window contributes `BOUNDS += [0x40, 0x41, 0x5B,
0x61]` with `MASKS += [0x00, 0x09, 0x00, 0x05]`. In the real table these
are interior entries: `BOUNDS` reads `…, 0x3A, 0x41, 0x5B, 0x61, 0x7B, …`
— the `0x41`, `0x5B`, `0x61`, `0x7B` boundaries in `bpe/unicode_tables.mojo`
are exactly the event points derived above.

**Step 5 — look up.** Two examples.

`_class_mask(0x47)` — codepoint `0x47`, which is `G`. The binary search
finds the greatest bound `<= 0x47`, which is `0x41`. The corresponding
mask is `0x09`, meaning `L` (`0x01`) and `u` (`0x08`) are set. So
`is_letter(0x47)` is true, `is_uppercase(0x47)` is true, and the other
predicates are false. That matches what you'd expect for `G`.

`_class_mask(0x5F)` — codepoint `0x5F`, which is `_` (underscore). The
greatest bound `<= 0x5F` is `0x5B`. Mask `0x00`. Every predicate is false.
That also matches — underscore isn't a letter, digit, mark, or whitespace.

The binary search takes about twelve probes on the real 3219-entry table,
which is negligible.

## 6. From the example to the real table

Everything above scales up mechanically. The differences are:

- **All six classes participate**, not three. Event points come from every
  interval of every class, plus `0` (prepended so codepoints below the
  first real boundary resolve to mask `0x00`).
- **Result: 3219 entries, 7 distinct masks, about 15.7 KB** (`3219 × (4 + 1)`
  bytes) for all ~1.1M codepoints — roughly a 350:1 compression ratio over
  the naive one-entry-per-codepoint design.
- **The last entry covers through `0x10FFFF`**, so every codepoint is in
  range. Surrogate codepoints (`0xD800–0xDFFF`) are excluded from the
  `regex` scan; whatever mask their segment carries is harmless because
  surrogates never appear in valid UTF-8.
- **The two derived predicates need no extra bits.** `is_upper_like = (L &
  ~l) | M` and `is_lower_like = (L & ~u) | M` are computed from the
  looked-up mask with a single mask load each. That's the win the o200k
  family gets over ~1,500 chained range comparisons in the original
  hand-written if-chains.
- **ASCII (`cp < 128`) never reaches the binary search** for five of the
  six base predicates. Hand-written fast paths answer directly, and the
  generator asserts those fast paths against the authoritative source on
  every run. (`is_whitespace` has no fast path — see §7 of the post for
  why.)

## 7. Why this is the right shape

The event-point construction isn't just a compression trick — it's the
natural representation of the underlying data. Unicode properties are
*preserved by runs*: long stretches of codepoints share the same
classification, and the changes between runs are few and well-defined.
Storing the runs directly, rather than storing per-codepoint values and
recovering the runs implicitly, means:

- **Storage is proportional to the number of changes, not to the number of
  codepoints.** If Unicode were 10× larger with the same density of
  property boundaries, the table would still be a few thousand entries.
- **Lookup is a single binary search.** All six class questions are
  answered by one probe, because the mask carries every class's bit.
- **The structure is checkable.** The generator can assert that the
  interval-derived table equals the authoritative `regex`-derived class
  definitions for every codepoint — a proof of equivalence, not a spot
  check.

That's why the event-point table is the right answer, and not just a
cleverer-but-equivalent alternative to the naive array.

## 8. File map and regeneration

| Piece | Where |
|---|---|
| Event-point collection + sweep + coalesce | `scripts/gen_unicode_tables.py` (`build_table`) |
| Per-class interval construction from `regex` | same file, `regex_intervals` + `main` |
| ASCII fast-path contract check | same file, `assert_ascii_branches` |
| Binary-search lookup + wrappers | `bpe/unicode_tables.mojo` (`_class_mask`, `is_*`) |
| Class model (why 6 bits suffice) | `scripts/gen_unicode_tables.py` (`BASE_FUNCS`) |
| Generator pipeline + acceptance | same file (`main`, `assert_ascii_branches`) |

Regenerate (idempotent — reproduces the committed table byte-for-byte):

```bash
pixi run --environment dev python scripts/gen_unicode_tables.py
