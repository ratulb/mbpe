#!/usr/bin/env python3
r"""Generate `bpe/unicode_tables.mojo` (event-point Unicode class table).

Authoritative source
--------------------
* Python `regex` module (dev/test-only, Unicode 16.0). This is the
  future-proof source used on every run.
* Historical note: this table replaces the literal if-chains formerly in
  `bpe/pretokenizer.mojo` (now deleted). The steady state is source-driven:
  every run derives the table from `regex` below.

The six class bits, one per `is_*` function (see UNICODE_TABLES.md):

    L = 0x01  \p{L}           (letters)
    N = 0x02  \p{N}           (numbers)
    l = 0x04  \p{Ll}          (lowercase letters)
    u = 0x08  \p{Lu}|\p{Lt}   (uppercase + titlecase letters)
    M = 0x10  \p{M}           (combining marks)
    W = 0x20  White_Space

How the whole script works
--------------------------
The pre-tokenizer must answer "is codepoint X a letter / digit / lowercase /
uppercase / mark / whitespace?" for all ~1.1M Unicode codepoints, exactly.
Storing one mask per codepoint would cost ~1.1 MB of mostly repetition:
long runs of codepoints share the same 6-bit mask, which only changes at a
few thousand positions. So this script builds an *event-point table*: one
entry per constant-mask run, as two parallel arrays (`BOUNDS`, run starts;
`MASKS`, the mask each run introduces). Mojo answers any query with a single
binary search over `BOUNDS` (see `UNICODE_EVENT_POINTS.md` for the worked
example). The pipeline, in order:

1. `regex_intervals` — ask the authoritative source (Python `regex`,
   Unicode 16.0) which codepoints belong to each of the six classes, as
   merged inclusive intervals. Surrogates (U+D800–U+DFFF) are excluded:
   they never appear in valid UTF-8, so whatever mask their segment
   carries is harmless.
2. `build_table` — collect every interval's `lo` and `hi + 1` as event
   points (plus 0), sweep them in order, and emit a new entry only where
   the mask actually changes (coalescing). Result: ~3,200 entries.
3. `assert_ascii_branches` — the generated Mojo answers `cp < 128` with
   handwritten fast paths instead of the binary search; this gate proves
   those branches equal the authoritative source below U+0080 on every
   run. (`is_whitespace` has no fast path, so it is skipped.)
4. `emit` — write `bpe/unicode_tables.mojo`: bit constants, the two
   arrays, the `_class_mask` binary search, the six `is_*` wrappers with
   their ASCII fast paths, and the three derived predicates
   (`is_letter_or_digit`, `is_upper_like`, `is_lower_like`), which are
   computed from a single mask load and need no extra table bits.

Regeneration is idempotent: re-running reproduces the committed table
byte-for-byte. On a Unicode bump, refresh the dev-env `regex` package and
re-run — never hand-edit `bpe/unicode_tables.mojo`.

Usage:
    pixi run --environment dev python scripts/gen_unicode_tables.py
"""

import sys
from bisect import bisect_right

MAX_CP = 0x110000  # one past U+10FFFF; range(MAX_CP) covers every codepoint
OUT = "bpe/unicode_tables.mojo"  # generated file: never hand-edit, re-run this script

# The six base classes. Each row is (Mojo function name, authoritative-source
# `regex` pattern, bit position in the 6-bit mask). The order here fixes the
# bit assignment used by both the table and the emitted wrappers.
BASE_FUNCS = (
    ("is_letter", r"\p{L}", 0x01),
    ("is_digit", r"\p{N}", 0x02),
    ("is_lowercase", r"\p{Ll}", 0x04),
    ("is_uppercase", r"(?:\p{Lu}|\p{Lt})", 0x08),
    ("is_mark", r"\p{M}", 0x10),
    ("is_whitespace", r"\p{White_Space}", 0x20),
)

# Verbatim ASCII fast paths + docstrings, part of the API contract. The
# generator asserts these against the authoritative source below U+0080 on
# every run (see assert_ascii_branches).
FUNC_META = {
    "is_letter": (
        "Exact Unicode property membership (generated from Unicode data).",
        "65 <= cp <= 90 or 97 <= cp <= 122",
    ),
    "is_digit": (
        "Exact Unicode property membership (generated from Unicode data).",
        "48 <= cp <= 57",
    ),
    "is_lowercase": (
        "Exact Unicode property membership (generated from Unicode data).",
        "97 <= cp <= 122",
    ),
    "is_uppercase": (
        "Exact Unicode property membership (generated from Unicode data).",
        "65 <= cp <= 90",
    ),
    "is_mark": (
        "Exact Unicode property membership (generated from Unicode data).",
        "False",
    ),
    # No ASCII fast path: the table is correct for every codepoint (including
    # U+000B/U+000C), and the pretokenizer's byte-level BYTE_CLASS LUT handles
    # ASCII before is_whitespace is ever reached.
    "is_whitespace": (
        "Return True if cp is a Unicode whitespace codepoint.",
        None,
    ),
}

BIT_L, BIT_N, BIT_l, BIT_u, BIT_M, BIT_W = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


# ── helpers ──────────────────────────────────────────────────────────────
# Small interval utilities. Everything here works on inclusive (lo, hi)
# codepoint ranges; merging keeps the per-class interval lists minimal so
# the binary searches in mask_at/_in_intervals stay cheap.

def merge_intervals(intervals):
    """Coalesce overlapping/adjacent inclusive (lo, hi) intervals."""
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def to_intervals(cps):
    """Turn an iterable of codepoints into merged inclusive intervals."""
    out = []
    for cp in sorted(cps):
        if out and cp == out[-1][1] + 1:
            out[-1] = (out[-1][0], cp)
        else:
            out.append((cp, cp))
    return out


def regex_intervals(pattern):
    """Codepoints matching `pattern`, as merged intervals (single pass).

    Scans every codepoint once through the authoritative `regex` tables and
    folds the matches into intervals. Surrogates are skipped: they are not
    valid UTF-8, so no query can ever depend on their mask.
    """
    import regex

    chars = [chr(cp) for cp in range(MAX_CP) if not (0xD800 <= cp <= 0xDFFF)]
    return to_intervals(ord(c) for c in regex.findall(pattern, "".join(chars)))


# ── table construction ───────────────────────────────────────────────────
# The event-point core (pipeline step 2). Turns per-class interval sets into
# the single (BOUNDS, MASKS) table the Mojo side binary-searches.

def build_table(class_intervals):
    """class_intervals: {name: [(lo, hi), ...]}. Returns (bounds, masks).

    Every inclusive interval (lo, hi) turns membership on at lo and off just
    after hi, so each contributes two event points: lo and hi + 1 (the +1
    converts inclusive ranges to half-open segments that tile the space with
    no gaps). Sweeping the sorted points and emitting only mask *changes*
    is what compresses ~1.1M codepoints into a few thousand entries.
    """
    los = {name: [lo for lo, hi in iv] for name, iv in class_intervals.items()}
    intervals = {name: iv for name, iv in class_intervals.items()}

    def mask_at(cp):
        # The mask at one point: OR in each class's bit if cp falls inside
        # one of that class's intervals (binary search per class via
        # bisect_right on the interval starts). Correct at any point, but
        # only *evaluated* at segment starts, where no class boundary can
        # fall inside the segment — so every codepoint in the segment
        # shares the result by construction.
        m = 0
        for name, _, bit in BASE_FUNCS:
            iv = intervals[name]
            i = bisect_right(los[name], cp) - 1
            if i >= 0 and iv[i][1] >= cp:
                m |= bit
        return m

    points = {0}  # 0 anchors the first segment: codepoints below the first
    # real boundary resolve to mask 0 instead of falling off the table.
    for iv in intervals.values():
        for lo, hi in iv:
            points.add(lo)
            points.add(hi + 1)
    points = sorted(points)

    bounds, masks, prev = [], [], None
    for a, b in zip(points, points[1:]):
        m = mask_at(a)
        if m != prev:  # coalesce: same mask as the previous segment means
            bounds.append(a)  # no new run starts here — skip the entry.
            masks.append(m)
            prev = m
    # The trailing segment (last point through end of space) has no right
    # neighbor to compare against, so it is handled explicitly: same
    # coalescing rule, with the run extending implicitly to U+10FFFF.
    m = mask_at(points[-1])
    if m != prev:
        bounds.append(points[-1])
        masks.append(m)

    return bounds, masks


def assert_ascii_branches(meta, classes):
    """Hardcoded ASCII fast paths must match the authoritative source below U+0080.

    Pipeline step 3 and the run's only correctness gate: the emitted Mojo
    short-circuits `cp < 128` with handwritten branches, bypassing the table
    on the hottest inputs. For each class, the set of codepoints below 128
    accepted by the branch expression must equal the set the authoritative
    intervals imply; any drift (e.g. someone edits a range) aborts the run
    instead of emitting a subtly wrong table.

    `classes` maps name -> merged intervals.  `is_whitespace` has no ASCII
    fast path (the table is correct for all codepoints), so it is skipped.
    """
    for name, _, _ in BASE_FUNCS:
        _doc, ascii_expr = meta[name]
        if ascii_expr is None:
            continue
        expected = {cp for cp in range(128)
                    if _in_intervals(cp, classes[name])}
        branch = {cp for cp in range(128)
                  if eval(ascii_expr, {"__builtins__": {}}, {"cp": cp})}
        if expected != branch:
            sys.exit(
                f"ASCII branch drift for {name}: authoritative source {sorted(expected)} "
                f"!= branch {sorted(branch)}"
            )
    print("ASCII fast paths == authoritative source below U+0080")


def _in_intervals(cp, iv):
    # Point query used by the ASCII gate: is cp inside any interval of the
    # merged list? bisect_right finds the last interval starting at or
    # before cp; cp is a member iff it does not end before cp.
    i = bisect_right([lo for lo, _ in iv], cp) - 1
    return i >= 0 and iv[i][1] >= cp


# ── emission ─────────────────────────────────────────────────────────────
# Pipeline step 4: render the Mojo source. The emitted `_class_mask` is a
# hand-rolled binary search for the greatest bound <= cp (the Mojo-side
# mirror of mask lookup); the `is_*` wrappers add the ASCII fast paths, and
# the derived predicates reuse one mask load instead of extra table bits.

def fmt_list(vals, per_line, hexfmt):
    lines = []
    for i in range(0, len(vals), per_line):
        chunk = vals[i : i + per_line]
        lines.append("    " + ", ".join(hexfmt(v) for v in chunk) + ",")
    return "\n".join(lines)


def emit(bounds, masks, meta):
    # Structural invariants first: the table must open at 0 (so every
    # codepoint has a run to land in) and its last run must extend through
    # U+10FFFF. Violations abort before anything is written.
    N = len(bounds)
    assert bounds[0] == 0, "table must start at U+0000"
    assert bounds[-1] <= MAX_CP - 1, "table must cover U+10FFFF"
    assert masks[0] == 0, "U+0000..first boundary must be mask 0"

    func_defs = []
    # One wrapper per base class: table lookup, plus the ASCII fast path
    # from FUNC_META where one exists (all but is_whitespace).
    for name, _, bit in BASE_FUNCS:
        _doc, ascii_expr = meta[name]
        if ascii_expr is None:
            func_defs.append(
                f"@always_inline\ndef {name}(cp: Int) -> Bool:\n"
                f"    return (_class_mask(UInt32(cp)) & UInt8({bit})) != 0\n"
            )
        else:
            func_defs.append(
                f"@always_inline\ndef {name}(cp: Int) -> Bool:\n"
                f"    if cp < 128:\n"
                f"        return {ascii_expr}\n"
                f"    return (_class_mask(UInt32(cp)) & UInt8({bit})) != 0\n"
            )
    func_defs.append(
        "@always_inline\ndef is_letter_or_digit(cp: Int) -> Bool:\n"
        "    return is_letter(cp) or is_digit(cp)\n"
    )
    # Derived predicates: no extra table bits. "Upper/lower-like" means "a
    # letter that isn't lowercase/uppercase, or a combining mark" — computed
    # from the already-loaded mask, with the same ASCII fast path treatment.
    func_defs.append(
        "@always_inline\ndef is_upper_like(cp: Int) -> Bool:\n"
        "    if cp < 128:\n"
        "        return 65 <= cp <= 90\n"
        "    var m = _class_mask(UInt32(cp))\n"
        "    return ((m & BIT_L) != 0 and (m & BIT_l) == 0) or (m & BIT_M) != 0\n"
    )
    func_defs.append(
        "@always_inline\ndef is_lower_like(cp: Int) -> Bool:\n"
        "    if cp < 128:\n"
        "        return 97 <= cp <= 122\n"
        "    var m = _class_mask(UInt32(cp))\n"
        "    return ((m & BIT_L) != 0 and (m & BIT_u) == 0) or (m & BIT_M) != 0\n"
    )

    distinct = sorted(set(masks))
    content = f"""\
from std.builtin.globals import global_constant
from std.collections.array import Array

comptime BIT_L: UInt8 = {BIT_L}
comptime BIT_N: UInt8 = {BIT_N}
comptime BIT_l: UInt8 = {BIT_l}
comptime BIT_u: UInt8 = {BIT_u}
comptime BIT_M: UInt8 = {BIT_M}
comptime BIT_W: UInt8 = {BIT_W}

comptime BOUNDS: Array[UInt32, {N}] = [
{fmt_list(bounds, 16, lambda v: f"0x{v:X}")}
]
comptime MASKS: Array[UInt8, {N}] = [
{fmt_list(masks, 32, lambda v: str(v))}
]

@always_inline
def _class_mask(cp: UInt32) -> UInt8:
    # 1.0.0: Array is no longer ImplicitlyCopyable, so a comptime table
    # cannot be touched directly at runtime (materialization error).
    # global_constant() parks each table in static memory once; the refs
    # below are zero-copy views (see the manual's "Global lookup tables").
    ref bounds = global_constant[BOUNDS]()
    ref masks = global_constant[MASKS]()
    var lo: Int = 0
    var hi: Int = {N - 1}
    while lo < hi:
        var mid = (lo + hi + 1) >> 1
        if bounds.unsafe_get(mid) <= cp:
            lo = mid
        else:
            hi = mid - 1
    return masks.unsafe_get(lo)

{chr(10).join(func_defs)}
"""
    with open(OUT, "w") as f:
        f.write(content)
    print(f"wrote {OUT} ({N} entries, {len(distinct)} distinct masks)")


# ── main ─────────────────────────────────────────────────────────────────
# Straight-line pipeline: derive intervals from the authoritative source,
# gate the ASCII contract, build the table, emit the Mojo file.

def unicode_note():
    import regex

    return f"16.0.0 (Python `regex` {regex.__version__} internal tables)"


def main():
    import regex  # fail fast here if the authoritative source is unavailable

    print(f"regex authoritative source: {unicode_note()}")
    classes = {}
    for name, pattern, _ in BASE_FUNCS:
        classes[name] = regex_intervals(pattern)
    assert_ascii_branches(FUNC_META, classes)

    bounds, masks = build_table(classes)

    emit(bounds, masks, FUNC_META)


if __name__ == "__main__":
    main()
