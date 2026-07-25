"""Benchmark comparing three encode approaches."""

from tokenizer import BPETokenizer, PreTokenizer
from std.time import perf_counter_ns


# ═══════════════════════════════════════════════════════════════════════════
# Shared: timer, formatting
# ═══════════════════════════════════════════════════════════════════════════

struct Timer:
    var _start: UInt
    def __init__(out self): self._start = 0
    def start(mut self): self._start = perf_counter_ns()
    def elapsed_ns(self) -> Int: return Int(perf_counter_ns() - self._start)

def min_ns(t: List[Int]) -> Int:
    var b = t[0]
    for i in range(1, len(t)):
        if t[i] < b: b = t[i]
    return b

def mean_ns(t: List[Int]) -> Int:
    var s: Int = 0
    for i in range(len(t)): s += t[i]
    return s // len(t)

def fmt_ns(ns: Int) -> String:
    if ns < 1000: return String(ns) + " ns"
    elif ns < 1_000_000: return String(ns // 1000) + "." + String((ns % 1000) // 100) + " us"
    elif ns < 1_000_000_000: return String(ns // 1_000_000) + "." + String((ns % 1_000_000) // 100_000) + " ms"
    else: return String(ns // 1_000_000_000) + "." + String((ns % 1_000_000_000) // 100_000_000) + " s"

def fmt_tok_s(tokens: Int, ns: Int) -> String:
    if ns == 0: return "N/A"
    var per_sec = tokens * 1_000_000_000 // ns
    if per_sec >= 1_000_000: return String(per_sec // 1_000_000) + "." + String((per_sec % 1_000_000) // 100_000) + " M tok/s"
    elif per_sec >= 1000: return String(per_sec // 1000) + "." + String((per_sec % 1000) // 100) + " K tok/s"
    else: return String(per_sec) + " tok/s"

def build_corpus() raises -> String:
    var sentences = List[String]()
    sentences.append(String("This is the Hugging Face Course."))
    sentences.append(String("This chapter is about tokenization."))
    sentences.append(String("This section shows several tokenizer algorithms."))
    sentences.append(String("Hopefully, you will be able to understand how they are trained and generate tokens."))
    var parts = List[String]()
    for _ in range(500):
        for i in range(len(sentences)):
            parts.append(sentences[i].copy())
    return String(" ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════
# Shared: raw token bytes storage (ID → raw UInt8 sequence)
# ═══════════════════════════════════════════════════════════════════════════

struct TokenBytes:
    var _data: List[UInt8]
    var _offsets: List[Int]
    var _lengths: List[Int]

    def __init__(out self):
        self._data = List[UInt8]()
        self._offsets = List[Int]()
        self._lengths = List[Int]()

    def build(mut self, tok: BPETokenizer) raises:
        for id in range(len(tok)):
            var s = tok.vocab[id]
            self._offsets.append(len(self._data))
            var count = 0
            for cp in s.codepoints():
                self._data.append(UInt8(tok.cp_to_byte[Int(cp)]))
                count += 1
            self._lengths.append(count)

    def get_bytes(self, id: Int, mut out: List[UInt8]):
        var off = self._offsets[id]
        var n = self._lengths[id]
        for i in range(n):
            out.append(self._data[off + i])


# ═══════════════════════════════════════════════════════════════════════════
# Approach B: RankTable (byte-span → rank)
# ═══════════════════════════════════════════════════════════════════════════

struct RankTable(Movable):
    var _fast: Dict[UInt64, Int32]
    var _overflow: Dict[String, Int32]

    def __init__(out self):
        self._fast = Dict[UInt64, Int32]()
        self._overflow = Dict[String, Int32]()

    def set(mut self, span: Span[UInt8, _], rank: Int32):
        if len(span) <= 8:
            var key: UInt64 = 0
            for i in range(len(span)):
                key = (key << 8) | UInt64(span[i])
            self._fast[key] = rank
        else:
            var chars = List[UInt8](capacity=len(span) * 2)
            var hex = String("0123456789abcdef")
            for i in range(len(span)):
                var b = Int(span[i])
                chars.append(hex.as_bytes()[b >> 4])
                chars.append(hex.as_bytes()[b & 0xF])
            self._overflow[String(from_utf8_lossy=Span[UInt8](chars))] = rank

    def lookup(self, span: Span[UInt8, _]) -> Int32:
        if len(span) <= 8:
            var key: UInt64 = 0
            for i in range(len(span)):
                key = (key << 8) | UInt64(span[i])
            return self._fast.get(key, -1)
        var chars = List[UInt8](capacity=len(span) * 2)
        var hex = String("0123456789abcdef")
        for i in range(len(span)):
            var b = Int(span[i])
            chars.append(hex.as_bytes()[b >> 4])
            chars.append(hex.as_bytes()[b & 0xF])
        return self._overflow.get(String(from_utf8_lossy=Span[UInt8](chars)), -1)


# ═══════════════════════════════════════════════════════════════════════════
# Approach C: PairCache (ID-pair → merged_id)
# ═══════════════════════════════════════════════════════════════════════════

comptime CACHE_SHIFT: Int = 10
comptime CACHE_SIZE: Int = 1000
comptime CACHE_ENTRIES: Int = 1 << (CACHE_SHIFT * 2)
comptime ENCODE_SHIFT: Int = 20

struct PairCache(Movable):
    var _cache: List[Int]
    var _fallback: Dict[Int, Int]

    def __init__(out self):
        self._cache = List[Int](length=CACHE_ENTRIES, fill=-1)
        self._fallback = Dict[Int, Int]()

    def set(mut self, id1: Int, id2: Int, merged_id: Int):
        if id1 < CACHE_SIZE and id2 < CACHE_SIZE:
            self._cache[(id1 << CACHE_SHIFT) | id2] = merged_id
        else:
            self._fallback[(id1 << ENCODE_SHIFT) | id2] = merged_id

    def get(self, id1: Int, id2: Int) -> Int:
        if id1 < CACHE_SIZE and id2 < CACHE_SIZE:
            return self._cache[(id1 << CACHE_SHIFT) | id2]
        return self._fallback.get((id1 << ENCODE_SHIFT) | id2, -1)


# ═══════════════════════════════════════════════════════════════════════════
# Build data structures from a trained tokenizer
# ═══════════════════════════════════════════════════════════════════════════

def build_rank_table(tok: BPETokenizer, ref bytes: TokenBytes) raises -> RankTable:
    var table = RankTable()
    for (a_id, b_id, merged_id) in tok.merges:
        var concat = List[UInt8]()
        bytes.get_bytes(a_id, concat)
        bytes.get_bytes(b_id, concat)
        table.set(Span[UInt8](concat), Int32(merged_id))
    return table^

def build_pair_cache(tok: BPETokenizer) raises -> PairCache:
    var cache = PairCache()
    for (a_id, b_id, merged_id) in tok.merges:
        cache.set(a_id, b_id, merged_id)
    return cache^


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

@always_inline
def _byte_ids(span: Span[UInt8, _]) -> List[Int]:
    var ids = List[Int](capacity=len(span))
    for i in range(len(span)):
        ids.append(Int(span[i]))
    return ids^

@always_inline
def _merge_inplace(mut ids: List[Int], a: Int, b: Int, m: Int):
    var n = len(ids)
    var w = 0
    var i = 0
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            ids[w] = m
            i += 2
        else:
            ids[w] = ids[i]
            i += 1
        w += 1
    ids.resize(w, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Per-word encode functions (all called from the same pre-tokenize loop)
# ═══════════════════════════════════════════════════════════════════════════

def _encode_word_baseline(tok: BPETokenizer, word_span: Span[UInt8, _]) raises -> List[Int]:
    """Encode a single word using the tokenizer's merge list (sequential)."""
    var ids = _byte_ids(word_span)
    for (a_id, b_id, merged_id) in tok.merges:
        _merge_inplace(ids, a_id, b_id, merged_id)
    return ids^

def _encode_word_ranked(tok: BPETokenizer, word_span: Span[UInt8, _],
                        ref rank: RankTable, ref bytes: TokenBytes) raises -> List[Int]:
    """Encode a single word using RankTable (byte-span → rank, greedy)."""
    var ids = _byte_ids(word_span)
    var n = len(ids)
    if n < 2:
        return ids^

    while True:
        var best_rank = Int32(-1)
        var best_i = -1
        for i in range(n - 1):
            var concat = List[UInt8]()
            bytes.get_bytes(ids[i], concat)
            bytes.get_bytes(ids[i + 1], concat)
            var r = rank.lookup(Span[UInt8](concat))
            if r >= 0 and (best_rank < 0 or r < best_rank):
                best_rank = r
                best_i = i

        if best_i < 0:
            break
        var a = ids[best_i]
        var b = ids[best_i + 1]
        _merge_inplace(ids, a, b, Int(best_rank))
        n = len(ids)

    return ids^

def _encode_word_paircache(tok: BPETokenizer, word_span: Span[UInt8, _],
                           ref cache: PairCache) raises -> List[Int]:
    """Encode a single word using PairCache (ID-pair → merged_id, greedy)."""
    var ids = _byte_ids(word_span)
    var n = len(ids)
    if n < 2:
        return ids^

    while n >= 2:
        var best_rank = -1
        var best_i = -1
        var best_a = -1
        var best_b = -1
        for i in range(n - 1):
            var merged = cache.get(ids[i], ids[i + 1])
            if merged >= 0 and (best_rank < 0 or merged < best_rank):
                best_rank = merged
                best_i = i
                best_a = ids[i]
                best_b = ids[i + 1]
        if best_i < 0:
            break
        _merge_inplace(ids, best_a, best_b, best_rank)
        n = len(ids)

    return ids^

# ═══════════════════════════════════════════════════════════════════════════
# Top-level encode: pre-tokenize, then dispatch per word
# ═══════════════════════════════════════════════════════════════════════════

def encode_baseline(tok: BPETokenizer, text: String) raises -> List[Int]:
    return tok.encode(text)

def encode_ranked(tok: BPETokenizer, text: String, ref rank: RankTable,
                  ref bytes: TokenBytes) raises -> List[Int]:
    var words = PreTokenizer.tokenize(text)
    var result = List[Int]()
    for word in words:
        var word_ids = _encode_word_ranked(tok, word.as_bytes(), rank, bytes)
        for id in word_ids:
            result.append(id)
    return result^

def encode_paircache(tok: BPETokenizer, text: String, ref cache: PairCache) raises -> List[Int]:
    var words = PreTokenizer.tokenize(text)
    var result = List[Int]()
    for word in words:
        var word_ids = _encode_word_paircache(tok, word.as_bytes(), cache)
        for id in word_ids:
            result.append(id)
    return result^


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════

def run() raises:
    print("=" * 55)
    print("  Encode approach comparison — simple_bpe")
    print("=" * 55)

    print("\nBuilding corpus and training...")
    var corpus = build_corpus()
    var tok = BPETokenizer()
    tok.train([corpus], 500)

    print("Building shared data structures...")
    var bytes = TokenBytes()
    bytes.build(tok)
    var rank = build_rank_table(tok, bytes)
    var cache = build_pair_cache(tok)

    var n_bytes = corpus.byte_length()
    print("\nCorpus: " + String(n_bytes) + " bytes")
    print("Vocab: " + String(len(tok)) + ", merges: " + String(len(tok.merges)))

    # Verify output consistency
    print("\nVerifying output consistency...")
    var sample = String("This is a test of the tokenizer.")
    var res_a = encode_baseline(tok, sample)
    var res_b = encode_ranked(tok, sample, rank, bytes)
    var res_c = encode_paircache(tok, sample, cache)

    var ok = (len(res_a) == len(res_b)) and (len(res_b) == len(res_c))
    if ok:
        for i in range(len(res_a)):
            if res_a[i] != res_b[i] or res_b[i] != res_c[i]:
                ok = False
                break
    print("  " + ("OK - all identical" if ok else "MISMATCH - outputs differ"))

    # Benchmark
    var timer = Timer()
    var n_iters = 100

    for approach_idx in range(3):
        var label: String
        if approach_idx == 0: label = "Approach A (Baseline - sequential rules)"
        elif approach_idx == 1: label = "Approach B (RankTable - byte-span lookup)"
        else: label = "Approach C (PairCache - ID-pair lookup)"
        print("\n  " + label)

        var times = List[Int]()
        var enc_tokens = 0
        for _ in range(n_iters):
            timer.start()
            if approach_idx == 0:
                var ids = encode_baseline(tok, corpus)
                enc_tokens = len(ids)
            elif approach_idx == 1:
                var ids = encode_ranked(tok, corpus, rank, bytes)
                enc_tokens = len(ids)
            else:
                var ids = encode_paircache(tok, corpus, cache)
                enc_tokens = len(ids)
            times.append(timer.elapsed_ns())

        var best = min_ns(times)
        var avg = mean_ns(times)
        print("    best: " + fmt_ns(best) + "  " + fmt_tok_s(enc_tokens, best))
        print("    mean: " + fmt_ns(avg) + "  " + fmt_tok_s(enc_tokens, avg))

    print("\n" + "=" * 55)
    print("  Benchmark complete")
    print("=" * 55)


def main() raises:
    run()
