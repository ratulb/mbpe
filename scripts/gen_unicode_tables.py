#!/usr/bin/env python3
r"""Generate `bpe/unicode_tables.mojo` (event-point Unicode class table).

Oracles
-------
* Primary: Python `regex` module (dev/test-only, Unicode 16.0). This is the
  future-proof source used on every run.
* Transition: the literal if-chains in `bpe/pretokenizer.mojo`. Parsed only
  while they still exist (i.e. before the swap); if present, their class
  intervals must match the `regex` oracle exactly, and the emitted table is
  exhaustively asserted bit-for-bit against them for all ~1.1M codepoints.

The six class bits, one per `is_*` function (see UNICODE_TABLES.md):

    L = 0x01  \p{L}           (letters)
    N = 0x02  \p{N}           (numbers)
    l = 0x04  \p{Ll}          (lowercase letters)
    u = 0x08  \p{Lu}|\p{Lt}   (uppercase + titlecase letters)
    M = 0x10  \p{M}           (combining marks)
    W = 0x20  White_Space

Usage:
    pixi run --environment dev python scripts/gen_unicode_tables.py [--no-regex]
"""

import argparse
import re
import sys
from bisect import bisect_right

MAX_CP = 0x110000
PRETOK = "bpe/pretokenizer.mojo"
OUT = "bpe/unicode_tables.mojo"

BASE_FUNCS = (
    ("is_letter", r"\p{L}", 0x01),
    ("is_digit", r"\p{N}", 0x02),
    ("is_lowercase", r"\p{Ll}", 0x04),
    ("is_uppercase", r"(?:\p{Lu}|\p{Lt})", 0x08),
    ("is_mark", r"\p{M}", 0x10),
    ("is_whitespace", r"\p{White_Space}", 0x20),
)

DERIVED = (
    ("is_letter_or_digit", "is_letter(cp) or is_digit(cp)"),
    ("is_upper_like", None),  # emitted with explicit boolean form
    ("is_lower_like", None),
)

# Verbatim ASCII fast paths + docstrings, part of the API contract (the swap
# must be byte-for-byte behavior-preserving).  Used directly once the chains
# are deleted; while the chains exist they are parsed and must equal these.
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
    # Note: U+000B/U+000C are White_Space but omitted here (pre-existing
    # quirk; the byte-level LUT handles them before is_whitespace is reached).
    "is_whitespace": (
        "Return True if cp is a Unicode whitespace codepoint.",
        "cp == 0x0009 or cp == 0x000A or cp == 0x000D or cp == 0x0020",
    ),
}

BIT_L, BIT_N, BIT_l, BIT_u, BIT_M, BIT_W = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
UPPER_LIKE = (BIT_L & ~BIT_l) | BIT_M  # (L & ~l) | M
LOWER_LIKE = (BIT_L & ~BIT_u) | BIT_M  # (L & ~u) | M


# ── helpers ──────────────────────────────────────────────────────────────

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


def parse_chain_body(body):
    """Return (docstring, ascii_expr, chain_intervals) from a function body."""
    m = re.search(r'"""([\s\S]*?)"""', body)
    doc = m.group(1) if m else ""
    rest = body[m.end():] if m else body

    ascii_expr = None
    m = re.search(r"if cp < 128:\n\s*return (.+)", rest)
    if m:
        ascii_expr = m.group(1).strip()

    chain = ""
    m = re.search(r"return \(\n([\s\S]*?)\n    \)", rest)
    if m:
        chain = m.group(1)

    intervals = []
    for lo, hi in re.findall(r"0x([0-9A-Fa-f]+) <= cp and cp <= 0x([0-9A-Fa-f]+)", chain):
        intervals.append((int(lo, 16), int(hi, 16)))
    for v in re.findall(r"cp == 0x([0-9A-Fa-f]+)", chain):
        intervals.append((int(v, 16), int(v, 16)))

    if ascii_expr is not None:
        ns = {}
        for cp in range(128):
            if eval(ascii_expr, {"__builtins__": {}}, {"cp": cp}):
                intervals.append((cp, cp))

    return doc, ascii_expr, merge_intervals(intervals)


def parse_chains(src):
    """Extract {name: (doc, ascii_expr, intervals)} for the base functions."""
    names = "|".join(name for name, _, _ in BASE_FUNCS)
    pat = re.compile(
        r"@always_inline\ndef (%s)\(cp: Int\) -> Bool:\n([\s\S]*?)(?=\n@always_inline\ndef |\n@always_inline\ndef|\Z)"
        % names
    )
    out = {}
    for m in pat.finditer(src):
        name, body = m.group(1), m.group(2)
        out[name] = parse_chain_body(body)
    return out


def regex_intervals(pattern):
    """Codepoints matching `pattern`, as merged intervals (single pass)."""
    import regex

    chars = [chr(cp) for cp in range(MAX_CP) if not (0xD800 <= cp <= 0xDFFF)]
    return to_intervals(ord(c) for c in regex.findall(pattern, "".join(chars)))


# ── table construction ───────────────────────────────────────────────────

def build_table(class_intervals):
    """class_intervals: {name: [(lo, hi), ...]}. Returns (bounds, masks)."""
    los = {name: [lo for lo, hi in iv] for name, iv in class_intervals.items()}
    intervals = {name: iv for name, iv in class_intervals.items()}

    def mask_at(cp):
        m = 0
        for name, _, bit in BASE_FUNCS:
            iv = intervals[name]
            i = bisect_right(los[name], cp) - 1
            if i >= 0 and iv[i][1] >= cp:
                m |= bit
        return m

    points = {0}
    for iv in intervals.values():
        for lo, hi in iv:
            points.add(lo)
            points.add(hi + 1)
    points = sorted(points)

    bounds, masks, prev = [], [], None
    for a, b in zip(points, points[1:]):
        m = mask_at(a)
        if m != prev:
            bounds.append(a)
            masks.append(m)
            prev = m
    m = mask_at(points[-1])
    if m != prev:
        bounds.append(points[-1])
        masks.append(m)

    return bounds, masks


def assert_equivalence(bounds, masks, chain, classes):
    """Exhaustively assert table bits == chain results for all codepoints."""
    def mask_of(cp):
        return masks[bisect_right(bounds, cp) - 1]

    def in_class(name, cp):
        iv = chain[name][2]
        i = bisect_right([lo for lo, _ in iv], cp) - 1
        return i >= 0 and iv[i][1] >= cp

    for cp in range(MAX_CP):
        m = mask_of(cp)
        for name, _, bit in BASE_FUNCS:
            if bool(m & bit) != in_class(name, cp):
                sys.exit(
                    f"MISMATCH at U+{cp:04X}: {name} chain={in_class(name, cp)} "
                    f"table={bool(m & bit)}"
                )
        old_upper = (in_class("is_letter", cp) and not in_class("is_lowercase", cp)) or in_class("is_mark", cp)
        table_upper = (bool(m & BIT_L) and not bool(m & BIT_l)) or bool(m & BIT_M)
        if table_upper != old_upper:
            sys.exit(f"MISMATCH is_upper_like at U+{cp:04X}")
        old_lower = (in_class("is_letter", cp) and not in_class("is_uppercase", cp)) or in_class("is_mark", cp)
        table_lower = (bool(m & BIT_L) and not bool(m & BIT_u)) or bool(m & BIT_M)
        if table_lower != old_lower:
            sys.exit(f"MISMATCH is_lower_like at U+{cp:04X}")
    print(f"equivalence asserted over all {MAX_CP} codepoints")


def assert_ascii_branches(meta, classes):
    """Hardcoded ASCII fast paths must match the oracle below U+0080.

    `classes` maps name -> merged intervals.  The one documented exception is
    is_whitespace: U+000B/U+000C are White_Space in the oracle but omitted
    from the ASCII branch (pre-existing quirk, preserved).
    """
    for name, _, _ in BASE_FUNCS:
        doc, ascii_expr = meta[name]
        ns = {}
        oracle = {cp for cp in range(128)
                  if _in_intervals(cp, classes[name])}
        branch = {cp for cp in range(128)
                  if eval(ascii_expr, {"__builtins__": {}}, {"cp": cp})}
        if name == "is_whitespace":
            branch |= {0x0B, 0x0C}  # quirk: oracle sees them as ws
        if oracle != branch:
            sys.exit(
                f"ASCII branch drift for {name}: oracle {sorted(oracle)} "
                f"!= branch {sorted(branch)}"
            )
    print("ASCII fast paths == oracle below U+0080 (quirk preserved)")


def _in_intervals(cp, iv):
    i = bisect_right([lo for lo, _ in iv], cp) - 1
    return i >= 0 and iv[i][1] >= cp


# ── emission ─────────────────────────────────────────────────────────────

def fmt_list(vals, per_line, hexfmt):
    lines = []
    for i in range(0, len(vals), per_line):
        chunk = vals[i : i + per_line]
        lines.append("    " + ", ".join(hexfmt(v) for v in chunk) + ",")
    return "\n".join(lines)


def emit(bounds, masks, meta, unicode_note):
    N = len(bounds)
    assert bounds[0] == 0, "table must start at U+0000"
    assert bounds[-1] <= MAX_CP - 1, "table must cover U+10FFFF"
    assert masks[0] == 0, "U+0000..first boundary must be mask 0"

    func_defs = []
    for name, _, bit in BASE_FUNCS:
        _doc, ascii_expr = meta[name]
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
from std.collections.inline_array import InlineArray

comptime BIT_L: UInt8 = {BIT_L}
comptime BIT_N: UInt8 = {BIT_N}
comptime BIT_l: UInt8 = {BIT_l}
comptime BIT_u: UInt8 = {BIT_u}
comptime BIT_M: UInt8 = {BIT_M}
comptime BIT_W: UInt8 = {BIT_W}

comptime BOUNDS: InlineArray[UInt32, {N}] = [
{fmt_list(bounds, 16, lambda v: f"0x{v:X}")}
]
comptime MASKS: InlineArray[UInt8, {N}] = [
{fmt_list(masks, 32, lambda v: str(v))}
]

@always_inline
def _class_mask(cp: UInt32) -> UInt8:
    var lo: Int = 0
    var hi: Int = {N - 1}
    while lo < hi:
        var mid = (lo + hi + 1) >> 1
        if BOUNDS.unsafe_get(mid) <= cp:
            lo = mid
        else:
            hi = mid - 1
    return MASKS.unsafe_get(lo)

{chr(10).join(func_defs)}
"""
    with open(OUT, "w") as f:
        f.write(content)
    print(f"wrote {OUT} ({N} entries, {len(distinct)} distinct masks)")


# ── main ─────────────────────────────────────────────────────────────────

def unicode_note():
    import regex

    return f"16.0.0 (Python `regex` {regex.__version__} internal tables)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-regex", action="store_true",
                    help="skip the regex oracle / chain-regex equality check")
    args = ap.parse_args()

    try:
        src = open(PRETOK).read()
    except FileNotFoundError:
        src = ""

    chain = parse_chains(src) if src else {}

    if not args.no_regex:
        import regex

        print(f"regex oracle: {unicode_note()}")
        classes = {}
        for name, pattern, _ in BASE_FUNCS:
            classes[name] = regex_intervals(pattern)
        if chain:
            for name, _, _ in BASE_FUNCS:
                if chain[name][2] != classes[name]:
                    sys.exit(
                        f"chain != regex oracle for {name} -- aborting "
                        f"(chain {len(chain[name][2])} intervals, "
                        f"regex {len(classes[name])} intervals)"
                    )
            print("chain == regex oracle for all 6 base classes")
        else:
            assert_ascii_branches(FUNC_META, classes)
    elif chain:
        classes = {name: chain[name][2] for name, _, _ in BASE_FUNCS}
    else:
        sys.exit("no chains and --no-regex: nothing to build from")

    if chain and len(chain) != len(BASE_FUNCS):
        sys.exit(f"could not parse all base functions; got {list(chain)}")

    bounds, masks = build_table(classes)

    if chain:
        # While the chains exist, their parsed (doc, ascii) must equal the
        # hardcoded contract, and the table is asserted against the chains.
        for name, _, _ in BASE_FUNCS:
            if chain[name][:2] != FUNC_META[name]:
                sys.exit(f"chain doc/ascii drift for {name}: {chain[name][:2]}")
        assert_equivalence(bounds, masks, chain, classes)
        meta = {name: FUNC_META[name] for name, _, _ in BASE_FUNCS}
    else:
        meta = FUNC_META

    note = unicode_note() if not args.no_regex else "generated offline (--no-regex)"
    emit(bounds, masks, meta, note)


if __name__ == "__main__":
    main()
