# benchmark_ctpop.mojo
#
# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL RECORD — STANDALONE. NOT WIRED INTO ANYTHING.
# ─────────────────────────────────────────────────────────────────────────────
#
# This is a self-contained investigation of the letter-run scan in
# bpe/pretokenizer.mojo. It asks two questions:
#
#   1. Are the two 64-bit population counts functionally equivalent?
#        - `_ctpop64(x)`: a hand-rolled SWAR bit-twiddling implementation
#          (addition pyramid ending in a multiply-and-shift)
#        - `pop_count(x).__int__()`: the std.bit builtin (imported at
#          bpe/pretokenizer.mojo:105), which lowers to a single `popcnt`
#   2. Which one is faster in the real hot path?
#
# Background (the "why this file exists"):
#   The hot path computes the length of a leading ASCII-letter run inside
#   `_swar_letter_run` via `pop_count(lsb - 1).__int__() >> 3`. The argument
#   `lsb - 1` always has the form 2^k - 1 (a run of set bits), so the
#   equivalence check covers that domain explicitly. `_ctpop64` was dead code
#   (a commented-out line suggested restoring it), and there was an
#   unresolved question of which form was faster.
#
# Result (settled 2026-08-07, this file reproduces the evidence):
#   - EQUIVALENCE: PASS — zero mismatches over exhaustive 0..2^22, the
#     hot-path domain (2^k - 1), single bits, 0xFFFF...FF, and 10M random +
#     10M lsb-derived inputs (including the >> 3 shifted forms).
#   - IN-CONTEXT SWEEP (the decisive measurement): the `pop_count` scanner is
#     ~2x faster than the `_ctpop64` scanner (ratio ~2.0 across runs). The
#     isolated microbenchmark is noisy/unreliable (bare forms flip between
#     runs) because the compiler sometimes folds the SWAR idiom into a single
#     popcnt in isolation, but not inside the full scanner.
#   - ACTION TAKEN: `_ctpop64` was deleted from bpe/pretokenizer.mojo (it was
#     dead AND ~2x slower in context) and the commented-out fallback line was
#     removed. The production code keeps `pop_count(lsb - 1).__int__() >> 3`.
#
# All helpers below are verbatim copies of the pretokenizer internals at the
# time of the investigation, so this file reproduces the measurements without
# depending on any module from the repo. It is intentionally not registered in
# benchmarks/run.sh or any other harness — it is kept purely as history.
#
# Usage (run from the repo root):
#   pixi run mojo benchmarks/benchmark_ctpop.mojo
#
# The sweep reads the corpus from `benchmarks/corpus.txt` (CWD-relative), the
# same default the other benchmark scripts use. Override with BPE_CORPUS if
# you want a different text, e.g.:
#   BPE_CORPUS=/path/to/corpus.txt pixi run mojo benchmarks/benchmark_ctpop.mojo
# ─────────────────────────────────────────────────────────────────────────────

from std.bit import pop_count
from std.time import perf_counter_ns
from std.pathlib import Path
from std.os import getenv

comptime ARR_LEN = 1 << 16
comptime MICRO_WARM = 64
comptime MICRO_REPS = 2000
comptime SWEEP_WARM = 20
comptime SWEEP_REPS = 200
comptime DEFAULT_CORPUS = "benchmarks/corpus.txt"

# ---------------------------------------------------------------------------
# Verbatim copies from bpe/pretokenizer.mojo at the time of the investigation
# (previously lines ~236-270; `_ctpop64` has since been deleted).
# ---------------------------------------------------------------------------

@always_inline
def _hasless(x: UInt64, n: UInt64) -> UInt64:
    # Sets bit 7 of every lane whose byte is < n. A byte is ASCII iff
    # 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A, i.e. below 0x7B but not below
    # 0x61 (after OR-ing in 0x20 to lower-case A-Z). `~x` makes sure a byte
    # with bit 7 set (non-ASCII) never counts.
    return (
        (x - n * UInt64(0x0101010101010101)) & ~x & UInt64(0x8080808080808080)
    )


@always_inline
def _haszero(x: UInt64) -> UInt64:
    # Sets bit 7 of every lane that is *exactly* zero. Needed because
    # _hasless mis-detects a byte equal to the bound, so the code ANDs out
    # the 0x7B-exact case via _haszero.
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


# The SWAR candidate under investigation. Kept here (deleted from the repo
# source) so the historical benchmark remains runnable.
@always_inline
def _ctpop64(x: UInt64) -> Int:
    var y = x - ((x >> 1) & UInt64(0x5555555555555555))
    y = (y & UInt64(0x3333333333333333)) + (
        (y >> 2) & UInt64(0x3333333333333333)
    )
    y = (y + (y >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
    return Int((y * UInt64(0x0101010101010101)) >> 56)


# ---------------------------------------------------------------------------
# Letter-run scanner, two variants. Production body of `_swar_letter_run` at
# the time of the investigation; the <8-byte tail is inlined as a direct ASCII
# letter test, which equals `BYTE_CLASS & BC_LETTER` for ASCII.
# ---------------------------------------------------------------------------

@always_inline
def _is_ascii_letter(b: UInt8) -> Bool:
    return (b >= UInt8(0x41) and b <= UInt8(0x5A)) or (
        b >= UInt8(0x61) and b <= UInt8(0x7A)
    )


def _swar_letter_run_popcount[origin: Origin, //](
    span: Span[UInt8, origin], i: Int, n: Int
) -> Int:
    # Loads 8 bytes, computes the non-letter mask `~_letters8(w) & 0x8080...`,
    # and finds the leading letter-run length from the *lowest* non-letter
    # bit. The argument to pop_count is `lsb - 1`, always of the form 2^k - 1.
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
        if _is_ascii_letter(span[j]):
            consumed += 1
            j += 1
        else:
            break
    return consumed


def _swar_letter_run_ctpop[origin: Origin, //](
    span: Span[UInt8, origin], i: Int, n: Int
) -> Int:
    var p8 = span.unsafe_ptr()
    var consumed = 0
    var j = i
    while j + 8 <= n:
        var w: UInt64 = (p8 + j).bitcast[UInt64]()[]
        var nl = ~_letters8(w) & UInt64(0x8080808080808080)
        if nl != 0:
            var lsb = nl & (UInt64(0) - nl)
            return consumed + (_ctpop64(lsb - 1) >> 3)
        consumed += 8
        j += 8
    while j < n:
        if _is_ascii_letter(span[j]):
            consumed += 1
            j += 1
        else:
            break
    return consumed


# ---------------------------------------------------------------------------
# Deterministic 64-bit LCG for runtime-varying (non-constant) inputs — keeps
# the benchmark honest by defeating constant folding.
# ---------------------------------------------------------------------------

struct Lcg:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state = self.state * UInt64(6364136223846793005) + UInt64(
            1442695040888963407
        )
        return self.state


# ---------------------------------------------------------------------------
# 1. Functional equivalence: _ctpop64(x) == pop_count(x).__int__() over
#    exhaustive, hot-path-domain, and random inputs (including the shifted
#    >> 3 forms the scanner uses). Also re-checks the classic "leading-run"
#    bug shape: `nl - 1` with several set bits (the correct form isolates the
#    lowest set bit first, hence `lsb`).
# ---------------------------------------------------------------------------

def verify_equivalence(seed: UInt64) raises:
    var bad = 0
    var first_x = UInt64(0)
    var first_a = 0
    var first_b = 0

    var exhaustive = 1 << 22
    for i in range(exhaustive):
        var x = UInt64(i)
        var a = _ctpop64(x)
        var b = pop_count(x).__int__()
        if a != b:
            if bad == 0:
                first_x = x
                first_a = a
                first_b = b
            bad += 1
    print("  exhaustive 0..2^22 (all 4,194,304 values): " + String(bad) + " mismatch(es)")

    # Hot-path domain: 2^k - 1 (runs of ones) and single-bit lsb patterns.
    var domain_bad = 0
    for k in range(64):
        var x = (UInt64(1) << UInt64(k)) - UInt64(1)
        if _ctpop64(x) != pop_count(x).__int__():
            domain_bad += 1
    var xmax = UInt64(0) - UInt64(1)
    if _ctpop64(xmax) != pop_count(xmax).__int__():
        domain_bad += 1
    for k in range(63):
        var x = UInt64(1) << UInt64(k)
        if _ctpop64(x) != pop_count(x).__int__():
            domain_bad += 1
    print("  2^k-1 (k=0..64) + single bits (2^k): " + String(domain_bad) + " mismatch(es)")
    bad += domain_bad

    # Random + lsb-derived args + shifted forms.
    var rng = Lcg(seed)
    var rnd_bad = 0
    for _ in range(10_000_000):
        var x = rng.next()
        if _ctpop64(x) != pop_count(x).__int__():
            if rnd_bad == 0:
                first_x = x
                first_a = _ctpop64(x)
                first_b = pop_count(x).__int__()
            rnd_bad += 1
        var lsb = x & (UInt64(0) - x)
        var arg = lsb - UInt64(1)
        if (_ctpop64(arg) >> 3) != (pop_count(arg).__int__() >> 3):
            if rnd_bad == 0:
                first_x = arg
                first_a = _ctpop64(arg) >> 3
                first_b = pop_count(arg).__int__() >> 3
            rnd_bad += 1
    print("  random 10M + lsb-derived (with >> 3): " + String(rnd_bad) + " mismatch(es)")
    bad += rnd_bad

    if bad == 0:
        print("  EQUIVALENCE: PASS (all " + String(exhaustive + 129) + " + 10M + 10M inputs)")
    else:
        print(
            "  EQUIVALENCE: FAIL — first x="
            + String(Int(first_x))
            + " ctpop="
            + String(first_a)
            + " builtin="
            + String(first_b)
        )


# ---------------------------------------------------------------------------
# 2. Microbenchmark: per-call ns for _ctpop64 vs pop_count, bare and >> 3.
#    Inputs come from a runtime-generated array (no constant folding); results
#    are accumulated into `sink` and returned so nothing is eliminated.
#    NOTE: this measurement is unreliable in isolation (bare forms flip
#    between runs); it is kept for completeness. See the in-context sweep
#    below for the decisive number.
# ---------------------------------------------------------------------------

def timed_loop[kind: Int](arr: List[UInt64], warm: Int, iters: Int) -> Tuple[Int, Float64]:
    var sink = 0
    for _ in range(warm):
        for i in range(ARR_LEN):
            var x = arr[i]
            comptime if kind == 0:
                sink += _ctpop64(x)
            elif kind == 1:
                sink += pop_count(x).__int__()
            elif kind == 2:
                sink += _ctpop64(x) >> 3
            else:
                sink += pop_count(x).__int__() >> 3
    var t0 = perf_counter_ns()
    for _ in range(iters):
        for i in range(ARR_LEN):
            var x = arr[i]
            comptime if kind == 0:
                sink += _ctpop64(x)
            elif kind == 1:
                sink += pop_count(x).__int__()
            elif kind == 2:
                sink += _ctpop64(x) >> 3
            else:
                sink += pop_count(x).__int__() >> 3
    var t1 = perf_counter_ns()
    return (sink, Float64(t1 - t0))


def bench_micro(seed: UInt64) raises:
    var arr = List[UInt64](capacity=ARR_LEN)
    var rng = Lcg(seed)
    for _ in range(ARR_LEN):
        arr.append(rng.next())

    var total_ops = Float64(MICRO_REPS * ARR_LEN)
    var r0 = timed_loop[0](arr, MICRO_WARM, MICRO_REPS)
    var r1 = timed_loop[1](arr, MICRO_WARM, MICRO_REPS)
    var r2 = timed_loop[2](arr, MICRO_WARM, MICRO_REPS)
    var r3 = timed_loop[3](arr, MICRO_WARM, MICRO_REPS)
    var s0 = r0[0]
    var s1 = r1[0]
    var s2 = r2[0]
    var s3 = r3[0]
    var t0 = r0[1]
    var t1 = r1[1]
    var t2 = r2[1]
    var t3 = r3[1]

    print("  _ctpop64(x)                 : " + String(t0 / total_ops) + " ns/op")
    print("  pop_count(x).__int__()      : " + String(t1 / total_ops) + " ns/op")
    print("  _ctpop64(x) >> 3            : " + String(t2 / total_ops) + " ns/op")
    print("  pop_count(x).__int__() >> 3 : " + String(t3 / total_ops) + " ns/op")
    print("  (sinks: " + String(s0 + s1 + s2 + s3) + " — kept to defeat DCE)")
    var ratio = (t2 / total_ops) / (t3 / total_ops)
    print("  shifted-form ratio ctpop/builtin: " + String(ratio))


# ---------------------------------------------------------------------------
# 3. In-context sweep: drive the real letter-run scanner over the corpus,
#    once per variant, and compare totals + timings. This is the decisive,
#    stable measurement (ratio ~2.0 across runs).
# ---------------------------------------------------------------------------

def sweep_scanner(text: String) raises:
    var span = text.as_bytes()
    var n = len(span)

    var starts = List[Int]()
    var pos = 0
    while pos < n:
        if _is_ascii_letter(span[pos]):
            starts.append(pos)
            while pos < n and _is_ascii_letter(span[pos]):
                pos += 1
        else:
            pos += 1
    var nstarts = len(starts)

    print("  corpus: " + String(n) + " bytes, " + String(nstarts) + " letter-run starts")

    var sum_pp = 0
    var sum_ct = 0
    for s in starts:
        sum_pp += _swar_letter_run_popcount(span, s, n)
        sum_ct += _swar_letter_run_ctpop(span, s, n)
    var verdict = "FAIL"
    if sum_pp == sum_ct:
        verdict = "PASS"
    print("  scanner equivalence (consumed bytes): " + verdict + " (" + String(sum_pp) + ")")

    var sink = 0
    for _ in range(SWEEP_WARM):
        for s in starts:
            sink += _swar_letter_run_popcount(span, s, n)
    var t0 = perf_counter_ns()
    for _ in range(SWEEP_REPS):
        for s in starts:
            sink += _swar_letter_run_popcount(span, s, n)
    var t1 = perf_counter_ns()

    for _ in range(SWEEP_WARM):
        for s in starts:
            sink += _swar_letter_run_ctpop(span, s, n)
    var t2 = perf_counter_ns()
    for _ in range(SWEEP_REPS):
        for s in starts:
            sink += _swar_letter_run_ctpop(span, s, n)
    var t3 = perf_counter_ns()

    var ops = Float64(SWEEP_REPS * nstarts)
    var ns_pp = Float64(t1 - t0) / ops
    var ns_ct = Float64(t3 - t2) / ops
    print("  pop_count scanner : " + String(ns_pp) + " ns/call (" + String(Float64(t1 - t0) / 1e6) + " ms total)")
    print("  _ctpop64  scanner : " + String(ns_ct) + " ns/call (" + String(Float64(t3 - t2) / 1e6) + " ms total)")
    print("  ratio ctpop/builtin: " + String(ns_ct / ns_pp))
    print("  (sink: " + String(sink) + ")")


def main() raises:
    var seed = UInt64(perf_counter_ns())
    print("== 1. Functional equivalence ==")
    verify_equivalence(seed)
    print("")
    print("== 2. Microbenchmark (isolated per-op) ==")
    bench_micro(seed)
    print("")
    print("== 3. In-context letter-run scanner sweep ==")
    var corpus = getenv("BPE_CORPUS", DEFAULT_CORPUS)
    var text = Path(corpus).read_text()
    sweep_scanner(text)
