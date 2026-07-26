"""Profile decode steps."""
from tokenizer import BPETokenizer
from std.pathlib import Path
from std.time import perf_counter_ns


def main() raises:
    var text = Path("benchmarks/corpus.txt").read_text()
    var corpus = List[String]()
    corpus.append(String(text))
    var tok = BPETokenizer()
    print("Training...")
    tok.train(corpus, 500)
    var ids = tok.encode(text)

    # Warmup
    for _ in range(5):
        _ = tok.decode(Span[Int](ids))

    # Pass 1: sum lengths
    var t0 = perf_counter_ns()
    for _ in range(20):
        var total: Int = 0
        for id in ids:
            total += tok.token_offsets[id + 1] - tok.token_offsets[id]
    var t1 = perf_counter_ns()
    print("pass 1 (sum):   " + String(Float64(t1 - t0) / 1e6 / 20) + " ms")

    # Pass 2: build byte list only (no Ġ check)
    var t2 = perf_counter_ns()
    for _ in range(20):
        var total2: Int = 0
        for id2 in ids:
            total2 += tok.token_offsets[id2 + 1] - tok.token_offsets[id2]
        var out2 = List[UInt8](capacity=total2)
        for id2 in ids:
            var start2 = tok.token_offsets[id2]
            var end2 = tok.token_offsets[id2 + 1]
            for r2 in range(start2, end2):
                out2.append(tok.token_bytes[r2])
    var t3 = perf_counter_ns()
    print("pass 2 (copy):  " + String(Float64(t3 - t2) / 1e6 / 20) + " ms")

    # Pass 2b: build with Ġ→space
    var t4 = perf_counter_ns()
    for _ in range(20):
        var total3: Int = 0
        for id3 in ids:
            total3 += tok.token_offsets[id3 + 1] - tok.token_offsets[id3]
        var out3 = List[UInt8](capacity=total3)
        for id3 in ids:
            var start3 = tok.token_offsets[id3]
            var end3 = tok.token_offsets[id3 + 1]
            var r3 = start3
            while r3 < end3:
                if r3 + 1 < end3 and tok.token_bytes[r3] == 0xC4 and tok.token_bytes[r3 + 1] == 0xA0:
                    out3.append(UInt8(0x20))
                    r3 += 2
                else:
                    out3.append(tok.token_bytes[r3])
                    r3 += 1
    var t5 = perf_counter_ns()
    print("pass 2b (Ġ→sp): " + String(Float64(t5 - t4) / 1e6 / 20) + " ms")

    # Step 3: String(from_utf8)
    # Build the byte list once, then measure from_utf8
    var total4: Int = 0
    for id4 in ids:
        total4 += tok.token_offsets[id4 + 1] - tok.token_offsets[id4]
    var prebuilt = List[UInt8](capacity=total4)
    for id4 in ids:
        var start4 = tok.token_offsets[id4]
        var end4 = tok.token_offsets[id4 + 1]
        for r4 in range(start4, end4):
            prebuilt.append(tok.token_bytes[r4])
    var t6 = perf_counter_ns()
    for _ in range(20):
        _ = String(from_utf8=Span[UInt8](prebuilt))
    var t7 = perf_counter_ns()
    print("step 3 (utf8):  " + String(Float64(t7 - t6) / 1e6 / 20) + " ms")

    # Full decode
    var t8 = perf_counter_ns()
    for _ in range(20):
        _ = tok.decode(Span[Int](ids))
    var t9 = perf_counter_ns()
    print("full decode:    " + String(Float64(t9 - t8) / 1e6 / 20) + " ms")
