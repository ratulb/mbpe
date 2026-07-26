from tokenizer import BPETokenizer
from pretokenizer import GPreTokenizer, GPT2Pretokenizer, GPT4Pretokenizer, PreTokenizer
from std.pathlib import Path
from std.testing import assert_equal, assert_true, TestSuite
from std.base64 import b64decode

from std.python import Python


def _check_splits[PT: PreTokenizer](pt: PT, text: String, expected: List[String]) raises:
    var actual = pt.split(text)
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


# ── Level A: Pre-tokenizer split alignment ──────────────────────

def test_gpre_splits() raises:
    var text = String(
        "Hello world! Don't stop I'll be there 123 U.S.A. new\nline  \n\n \n"
    )
    var expected = List[String]()
    expected.append(String("Hello"))
    expected.append(String("Ġworld!"))
    expected.append(String("ĠDon't"))
    expected.append(String("Ġstop"))
    expected.append(String("ĠI'll"))
    expected.append(String("Ġbe"))
    expected.append(String("Ġthere"))
    expected.append(String("Ġ123"))
    expected.append(String("ĠU"))
    expected.append(String(".S"))
    expected.append(String(".A"))
    expected.append(String("."))
    expected.append(String("Ġnew\nline"))
    expected.append(String("Ġ"))
    expected.append(String("Ġ\n\n"))
    expected.append(String("Ġ\n"))
    _check_splits(GPreTokenizer(), text, expected)


def test_gpt2_splits() raises:
    var text = String(
        "Hello world! Don't stop I'll be there 123 U.S.A. new\nline  \n\n \n"
    )
    var expected = List[String]()
    expected.append(String("Hello"))
    expected.append(String(" world"))
    expected.append(String("!"))
    expected.append(String(" Don"))
    expected.append(String("'t"))
    expected.append(String(" stop"))
    expected.append(String(" I"))
    expected.append(String("'ll"))
    expected.append(String(" be"))
    expected.append(String(" there"))
    expected.append(String(" 123"))
    expected.append(String(" U"))
    expected.append(String("."))
    expected.append(String("S"))
    expected.append(String("."))
    expected.append(String("A"))
    expected.append(String("."))
    expected.append(String(" new"))
    expected.append(String("\n"))
    expected.append(String("line"))
    expected.append(String("  \n\n \n"))
    _check_splits(GPT2Pretokenizer(), text, expected)


def test_gpt4_splits() raises:
    var text = String(
        "Hello world! Don't stop I'll be there 123 U.S.A. new\nline  \n\n \n"
    )
    var expected = List[String]()
    expected.append(String("Hello"))
    expected.append(String(" world"))
    expected.append(String("!"))
    expected.append(String(" Don"))
    expected.append(String("'t"))
    expected.append(String(" stop"))
    expected.append(String(" I"))
    expected.append(String("'ll"))
    expected.append(String(" be"))
    expected.append(String(" there"))
    expected.append(String(" "))
    expected.append(String("123"))
    expected.append(String(" U"))
    expected.append(String(".S"))
    expected.append(String(".A"))
    expected.append(String("."))
    expected.append(String(" new"))
    expected.append(String("\n"))
    expected.append(String("line"))
    expected.append(String("  \n\n \n"))
    _check_splits(GPT4Pretokenizer(), text, expected)


def test_split_counts() raises:
    """Verify split counts match Python regex reference on full corpus."""
    var text = Path("benchmarks/corpus.txt").read_text()
    assert_equal(len(GPreTokenizer().split(text)), 179425)
    assert_equal(len(GPT2Pretokenizer().split(text)), 265727)
    assert_equal(len(GPT4Pretokenizer().split(text)), 242095)


def test_byte_level_no_unk() raises:
    var corpus = List[String]()
    corpus.append(String("abc"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var ids = tok.encode(String("xyz"))
    assert_equal(len(ids), 3)
    assert_true(ids[0] != 0)
    assert_true(ids[1] != 0)
    assert_true(ids[2] != 0)
    assert_equal(tok.decode(ids), "xyz")


def test_basic_roundtrip() raises:
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    assert_true(len(tok) >= 256)

    var ids = tok.encode(String("hello world"))
    assert_true(len(ids) > 0)
    assert_equal(tok.decode(ids), "hello world")


def test_empty_input() raises:
    var corpus = List[String]()
    corpus.append(String("abc"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    assert_equal(len(tok.encode(String(""))), 0)
    assert_equal(tok.decode(List[Int]()), "")


def test_save_load() raises:
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    tok.save("/tmp/bpe_test.json")
    var loaded = BPETokenizer.load("/tmp/bpe_test.json")

    assert_equal(len(loaded), len(tok))
    assert_equal(
        loaded.decode(loaded.encode(String("hello world"))),
        "hello world",
    )


def test_deterministic() raises:
    var corpus = List[String]()
    corpus.append(String("hello world"))

    var tok1 = BPETokenizer()
    tok1.train(corpus, 300)
    var tok2 = BPETokenizer()
    tok2.train(corpus, 300)

    assert_equal(len(tok1.merges), len(tok2.merges))


def test_full_hf_corpus() raises:
    var corpus = List[String]()
    corpus.append(String("This is the Hugging Face Course."))
    corpus.append(String("This chapter is about tokenization."))
    corpus.append(String("This section shows several tokenizer algorithms."))
    corpus.append(String(
        "Hopefully, you will be able to understand how they are trained and"
        " generate tokens."
    ))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    assert_true(len(tok) >= 256)
    assert_equal(
        tok.decode(tok.encode(String("This is not a token."))),
        "This is not a token.",
    )

    tok.save("/tmp/bpe_hf_test.json")
    var loaded = BPETokenizer.load("/tmp/bpe_hf_test.json")
    assert_equal(len(loaded), len(tok))
    assert_equal(
        loaded.decode(loaded.encode(String("This is not a token."))),
        "This is not a token.",
    )


def test_save_tiktoken() raises:
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    var path = "/tmp/bpe_tiktoken_test.tiktoken"
    tok.save_tiktoken(path)
    var content = Path(path).read_text()
    assert_true(content.byte_length() > 0)

    # Verify first 256 lines are single bytes (base64 of single byte)
    var lines = content.split("\n")
    assert_true(len(lines) >= 257)  # 256 bytes + at least 1 merge token
    assert_true(lines[0].find(" ") >= 0)


def test_tiktoken_roundtrip() raises:
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    var test_input = String("hello world")
    var original_ids = tok.encode(test_input)

    var path = "/tmp/bpe_tiktoken_rt.tiktoken"
    tok.save_tiktoken(path)

    var loaded = BPETokenizer()
    loaded.load_tiktoken(path)

    assert_equal(len(loaded), len(tok))
    assert_equal(len(loaded.merges), len(tok.merges))
    assert_equal(loaded.decode(loaded.encode(test_input)), test_input)

    var loaded_ids = loaded.encode(test_input)
    assert_equal(len(loaded_ids), len(original_ids))
    for i in range(len(original_ids)):
        assert_equal(loaded_ids[i], original_ids[i])


# ═══════════════════════════════════════════════════════════════
# Level B — .tiktoken file format structural integrity
# ═══════════════════════════════════════════════════════════════

def test_tiktoken_structure() raises:
    """Verify every .tiktoken line is valid base64+rank, first 256 lines
       decode to single bytes 0x00–0xFF, merge tokens are multi-byte,
       and ranks are strictly ascending."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var path = "/tmp/bpe_struct_test.tiktoken"
    tok.save_tiktoken(path)
    var content = Path(path).read_text()
    var raw_lines = content.split("\n")
    # Filter empty trailing lines.
    var lines = List[String]()
    for l in raw_lines:
        var s = String(l.strip())
        if s.byte_length() > 0:
            lines.append(s)
    # Expect at least 257 lines (256 base bytes + ≥1 merge token).
    assert_true(len(lines) >= 257)
    # First 256 lines = single-byte base tokens.
    for i in range(256):
        var parts = lines[i].split(" ")
        assert_equal(len(parts), 2)
        var decoded = b64decode(parts[0])
        assert_equal(len(decoded), 1)
        var rank = Int(parts[1])
        assert_true(rank >= 0)
    # Remaining lines = merge tokens (≥2 bytes each).
    for i in range(256, len(lines)):
        var parts = lines[i].split(" ")
        assert_equal(len(parts), 2)
        var decoded = b64decode(parts[0])
        assert_true(len(decoded) >= 2)
    # Ranks must be strictly ascending.
    var prev_rank = -1
    for i in range(len(lines)):
        var parts = lines[i].split(" ")
        var rank = Int(parts[1])
        assert_true(rank > prev_rank)
        prev_rank = rank


def test_tiktoken_deterministic_save() raises:
    """Two saves from the same tokenizer produce bit-identical files."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var p1 = "/tmp/bpe_det1.tiktoken"
    var p2 = "/tmp/bpe_det2.tiktoken"
    tok.save_tiktoken(p1)
    tok.save_tiktoken(p2)
    var c1 = Path(p1).read_text()
    var c2 = Path(p2).read_text()
    assert_equal(c1.byte_length(), c2.byte_length())
    assert_equal(c1, c2)


# ═══════════════════════════════════════════════════════════════
# Level C — Merge recovery correctness
# ═══════════════════════════════════════════════════════════════

def test_tiktoken_merge_consistency() raises:
    """Every recovered merge satisfies vocab[merged] == vocab[left] + vocab[right].
       This validates that _recover_merges assigned the correct left/right IDs."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    tok.save_tiktoken("/tmp/bpe_mc_test.tiktoken")
    var loaded = BPETokenizer()
    loaded.load_tiktoken("/tmp/bpe_mc_test.tiktoken")
    assert_true(len(loaded.merges) > 0)
    for mr in loaded.merges:
        var left = loaded.vocab[mr.first].copy()
        var right = loaded.vocab[mr.second].copy()
        var expected = left + right
        assert_equal(loaded.vocab[mr.merged], expected)


# ═══════════════════════════════════════════════════════════════
# Level D — Idempotency & cross-format parity
# ═══════════════════════════════════════════════════════════════

def test_tiktoken_idempotent_save() raises:
    """Save → load → save → load: final tokenizer produces identical
       encode results as the original."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var test_input = String("hello world")
    var original_ids = tok.encode(test_input)

    tok.save_tiktoken("/tmp/bpe_idem1.tiktoken")
    var l1 = BPETokenizer()
    l1.load_tiktoken("/tmp/bpe_idem1.tiktoken")
    l1.save_tiktoken("/tmp/bpe_idem2.tiktoken")
    var l2 = BPETokenizer()
    l2.load_tiktoken("/tmp/bpe_idem2.tiktoken")

    var final_ids = l2.encode(test_input)
    assert_equal(len(final_ids), len(original_ids))
    for i in range(len(original_ids)):
        assert_equal(final_ids[i], original_ids[i])
    assert_equal(l2.decode(final_ids), test_input)


def test_tiktoken_vs_json_parity() raises:
    """JSON save/load and .tiktoken save/load from the same trained
       tokenizer produce identical encode results."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var test_input = String("hello world")

    tok.save("/tmp/bpe_json_parity.json")
    tok.save_tiktoken("/tmp/bpe_tk_parity.tiktoken")

    var from_json = BPETokenizer.load("/tmp/bpe_json_parity.json")
    var from_tiktoken = BPETokenizer()
    from_tiktoken.load_tiktoken("/tmp/bpe_tk_parity.tiktoken")

    assert_equal(len(from_json), len(from_tiktoken))
    var json_ids = from_json.encode(test_input)
    var tk_ids = from_tiktoken.encode(test_input)
    assert_equal(len(json_ids), len(tk_ids))
    for i in range(len(json_ids)):
        assert_equal(json_ids[i], tk_ids[i])
    assert_equal(from_json.decode(tk_ids), test_input)


# ═══════════════════════════════════════════════════════════════
# Level E — Multi-pre-tokenizer roundtrip
# ═══════════════════════════════════════════════════════════════

def test_tiktoken_gpre_roundtrip() raises:
    """Full .tiktoken roundtrip with GPreTokenizer (explicit)."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer[GPreTokenizer]()
    tok.train(corpus, 300)
    var test_input = String("hello world")
    var original_ids = tok.encode(test_input)

    tok.save_tiktoken("/tmp/bpe_gpre_rt.tiktoken")
    var loaded = BPETokenizer[GPreTokenizer]()
    loaded.load_tiktoken("/tmp/bpe_gpre_rt.tiktoken")

    assert_equal(len(loaded.merges), len(tok.merges))
    var loaded_ids = loaded.encode(test_input)
    assert_equal(len(loaded_ids), len(original_ids))
    for i in range(len(original_ids)):
        assert_equal(loaded_ids[i], original_ids[i])
    assert_equal(loaded.decode(loaded_ids), test_input)


def test_tiktoken_gpt2_roundtrip() raises:
    """Full .tiktoken roundtrip with GPT2Pretokenizer."""
    var corpus = List[String]()
    corpus.append(String("Hello world! Don't stop."))
    var tok = BPETokenizer[GPT2Pretokenizer]()
    tok.train(corpus, 300)
    var test_input = String("Hello world! Don't stop.")
    var original_ids = tok.encode(test_input)

    tok.save_tiktoken("/tmp/bpe_gpt2_rt.tiktoken")
    var loaded = BPETokenizer[GPT2Pretokenizer]()
    loaded.load_tiktoken("/tmp/bpe_gpt2_rt.tiktoken")

    assert_equal(len(loaded.merges), len(tok.merges))
    var loaded_ids = loaded.encode(test_input)
    assert_equal(len(loaded_ids), len(original_ids))
    for i in range(len(original_ids)):
        assert_equal(loaded_ids[i], original_ids[i])
    assert_equal(loaded.decode(loaded_ids), test_input)


def test_tiktoken_gpt4_roundtrip() raises:
    """Full .tiktoken roundtrip with GPT4Pretokenizer."""
    var corpus = List[String]()
    corpus.append(String("Hello world! Don't stop."))
    var tok = BPETokenizer[GPT4Pretokenizer]()
    tok.train(corpus, 300)
    var test_input = String("Hello world! Don't stop.")
    var original_ids = tok.encode(test_input)

    tok.save_tiktoken("/tmp/bpe_gpt4_rt.tiktoken")
    var loaded = BPETokenizer[GPT4Pretokenizer]()
    loaded.load_tiktoken("/tmp/bpe_gpt4_rt.tiktoken")

    assert_equal(len(loaded.merges), len(tok.merges))
    var loaded_ids = loaded.encode(test_input)
    assert_equal(len(loaded_ids), len(original_ids))
    for i in range(len(original_ids)):
        assert_equal(loaded_ids[i], original_ids[i])
    assert_equal(loaded.decode(loaded_ids), test_input)


# ═══════════════════════════════════════════════════════════════
# Level F — Edge cases: empty corpus, Unicode text, minimal vocab
# ═══════════════════════════════════════════════════════════════

def test_tiktoken_empty_corpus() raises:
    """Save/load tokenizer trained on minimal data (single char)
       and verify encode/decode roundtrip works."""
    var corpus = List[String]()
    corpus.append(String("a"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    tok.save_tiktoken("/tmp/bpe_ec_test.tiktoken")
    var loaded = BPETokenizer()
    loaded.load_tiktoken("/tmp/bpe_ec_test.tiktoken")

    var encoded = loaded.encode(String("abc"))
    assert_true(len(encoded) >= 3)
    assert_equal(loaded.decode(encoded), "abc")

    encoded = loaded.encode(String(""))
    assert_equal(len(encoded), 0)
    assert_equal(loaded.decode(encoded), "")


def test_tiktoken_unicode_text() raises:
    """Roundtrip text containing multi-byte UTF-8 characters through
       .tiktoken save/load."""
    var corpus = List[String]()
    corpus.append(String("héllo wörld café"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var test_input = String("café wörld")
    var original_ids = tok.encode(test_input)

    tok.save_tiktoken("/tmp/bpe_uni_test.tiktoken")
    var loaded = BPETokenizer()
    loaded.load_tiktoken("/tmp/bpe_uni_test.tiktoken")

    assert_equal(len(loaded), len(tok))
    assert_equal(len(loaded.merges), len(tok.merges))
    var loaded_ids = loaded.encode(test_input)
    assert_equal(len(loaded_ids), len(original_ids))
    for i in range(len(original_ids)):
        assert_equal(loaded_ids[i], original_ids[i])
    assert_equal(loaded.decode(loaded_ids), "café wörld")


def test_tiktoken_no_merges() raises:
    """Save/load a tokenizer with zero merges (vocab_size=256).
       Only base bytes, no merge tokens.  Verify encode roundtrip."""
    var corpus = List[String]()
    corpus.append(String("hello"))
    var tok = BPETokenizer()
    tok.train(corpus, 256)
    assert_equal(len(tok), 256)
    assert_equal(len(tok.merges), 0)

    tok.save_tiktoken("/tmp/bpe_nm_test.tiktoken")
    var loaded = BPETokenizer()
    loaded.load_tiktoken("/tmp/bpe_nm_test.tiktoken")

    assert_equal(len(loaded), 256)
    assert_equal(len(loaded.merges), 0)
    assert_equal(loaded.decode(loaded.encode(String("hello"))), "hello")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
