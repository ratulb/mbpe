from bpe.tokenizer import BPETokenizer
from bpe.pretokenizer import GPreTokenizer, GPT2Pretokenizer, GPT4Pretokenizer, PreTokenizer, ByteMapping
from std.pathlib import Path
from std.testing import assert_equal, assert_true, TestSuite
from std.base64 import b64decode

from std.python import Python


def check_splits[PT: PreTokenizer](pt: PT, text: String, expected: List[String]) raises:
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
    check_splits(GPreTokenizer(), text, expected)


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
    check_splits(GPT2Pretokenizer(), text, expected)


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
    check_splits(GPT4Pretokenizer[ByteMapping.SEQUENTIAL](), text, expected)


def test_split_counts() raises:
    """Verify split counts match Python regex reference on full corpus."""
    var text = Path("benchmarks/corpus.txt").read_text()
    assert_equal(len(GPreTokenizer().split(text)), 179425)
    assert_equal(len(GPT2Pretokenizer().split(text)), 265727)
    assert_equal(len(GPT4Pretokenizer[ByteMapping.SEQUENTIAL]().split(text)), 242095)


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

    tok.save_tiktoken("/tmp/bpe_hf_test.tiktoken")
    var loaded = BPETokenizer()
    loaded.load_tiktoken("/tmp/bpe_hf_test.tiktoken")
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


def test_tiktoken_save_load_roundtrip() raises:
    """.tiktoken save/load roundtrip preserves encode/decode."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var test_input = String("hello world")

    tok.save_tiktoken("/tmp/bpe_tk_rt.tiktoken")
    var loaded = BPETokenizer()
    loaded.load_tiktoken("/tmp/bpe_tk_rt.tiktoken")

    assert_equal(len(loaded), len(tok))
    var tk_ids = loaded.encode(test_input)
    assert_equal(loaded.decode(tk_ids), test_input)


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
    var tok = BPETokenizer[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]()
    tok.train(corpus, 300)
    var test_input = String("Hello world! Don't stop.")
    var original_ids = tok.encode(test_input)

    tok.save_tiktoken("/tmp/bpe_gpt4_rt.tiktoken")
    var loaded = BPETokenizer[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]()
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


# ═══════════════════════════════════════════════════════════════
# Step 8 — o200k_base interop test
# ═══════════════════════════════════════════════════════════════

def test_load_o200k_base() raises:
    """Load OpenAI's real o200k_base.tiktoken and verify structural correctness
    plus encode/decode roundtrip.  Uses GPT4Pretokenizer[ByteMapping.SHUFFLED]
    because o200k_base uses a permuted byte-to-ID mapping (rank 0 = '!' not 0x00)."""
    var tok = BPETokenizer[GPT4Pretokenizer[ByteMapping.SHUFFLED]]()
    tok.load_tiktoken("/home/tenmoomnet/bpe.mojo/data/o200k_base.tiktoken")

    # o200k_base has 199,998 file entries, plus 21 reserved/special IDs (199998-200018)
    assert_equal(len(tok), 200019)
    assert_equal(len(tok.special_bytes), 2)
    assert_equal(tok.special_bytes["<|endoftext|>"], 199999)
    assert_equal(tok.special_bytes["<|endofprompt|>"], 200018)

    # Should have 199,742 merges (199998 - 256 base bytes)
    assert_equal(len(tok.merges), 199742)

    # Verify the loaded tokenizer can be saved and re-loaded
    var path = "/tmp/bpe_o200k_roundtrip.tiktoken"
    tok.save_tiktoken(path)
    var reloaded = BPETokenizer[GPT4Pretokenizer[ByteMapping.SHUFFLED]]()
    reloaded.load_tiktoken(path)
    assert_equal(len(reloaded), len(tok))
    assert_equal(len(reloaded.merges), len(tok.merges))

    # Verify that merge consistency holds: vocab[merged] == vocab[left] + vocab[right]
    var checked = 0
    for mr in reloaded.merges:
        var left = reloaded.vocab[mr.first].copy()
        var right = reloaded.vocab[mr.second].copy()
        var expected = left + right
        assert_equal(reloaded.vocab[mr.merged], expected)
        checked += 1
        if checked >= 1000:
            break

    # Verify encode/decode roundtrip — this was previously blocked because
    # o200k uses a shuffled byte mapping.  Now GPT4Pretokenizer[SHUFFLED]
    # handles the mapping correctly.
    var text = String("Hello world!")
    var ids = tok.encode(text)
    assert_true(len(ids) > 0, "o200k encode must produce tokens")
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


# ═══════════════════════════════════════════════════════════════
# Step 9 — Functional parity: train → save → load → encode match
# ═══════════════════════════════════════════════════════════════

def test_tiktoken_load_parity() raises:
    """Train a tokenizer, save as .tiktoken, load into a fresh instance,
    and verify encode produces identical token IDs.  This validates that
    load_tiktoken fully recovers the encoding behavior."""
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

    var test_input = String("This is not a token.")
    var original_ids = tok.encode(test_input)

    var path = "/tmp/bpe_parity_test.tiktoken"
    tok.save_tiktoken(path)

    var loaded = BPETokenizer()
    loaded.load_tiktoken(path)

    assert_equal(len(loaded), len(tok))
    assert_equal(len(loaded.merges), len(tok.merges))

    var loaded_ids = loaded.encode(test_input)
    assert_equal(len(loaded_ids), len(original_ids))
    for i in range(len(original_ids)):
        assert_equal(loaded_ids[i], original_ids[i])
    assert_equal(loaded.decode(loaded_ids), test_input)


# ═══════════════════════════════════════════════════════════════
# Byte mapping tests
# ═══════════════════════════════════════════════════════════════

def test_byte_mapping_sequential() raises:
    """For GPT4 SEQUENTIAL (cl100k), byte_to_id is identity for all 256 bytes."""
    for b in range(256):
        assert_equal(GPT4Pretokenizer[ByteMapping.SEQUENTIAL].byte_to_id(b), b)
        assert_equal(GPT4Pretokenizer[ByteMapping.SEQUENTIAL].id_to_byte(b), b)


def test_byte_mapping_shuffled() raises:
    """For GPT4 SHUFFLED (o200k), LUT permutes the byte→ID mapping.

    Spot-checks: ASCII letters (0x61-0x7A) map to ranks 64-89,
    space (0x20) maps to 220, DEL (0x7F) maps to 221.
    """
    # Space (0x20) → rank 220
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].byte_to_id(0x20), 220)
    # Rank 220 → space (0x20)
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].id_to_byte(220), 0x20)

    # DEL (0x7F) → rank 221
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].byte_to_id(0x7F), 221)
    # Rank 221 → DEL (0x7F)
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].id_to_byte(221), 0x7F)

    # 'a' (0x61) → rank 64
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].byte_to_id(0x61), 64)
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].id_to_byte(64), 0x61)

    # 'z' (0x7A) → rank 89
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].byte_to_id(0x7A), 89)
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].id_to_byte(89), 0x7A)

    # '!' (0x21) → rank 0 (lowest rank in o200k)
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].byte_to_id(0x21), 0)
    assert_equal(GPT4Pretokenizer[ByteMapping.SHUFFLED].id_to_byte(0), 0x21)

    # Inverse mapping: id_to_byte(byte_to_id(b)) == b for every byte
    for b in range(256):
        assert_equal(
            GPT4Pretokenizer[ByteMapping.SHUFFLED].id_to_byte(
                GPT4Pretokenizer[ByteMapping.SHUFFLED].byte_to_id(b)
            ),
            b,
        )


def test_byte_mapping_roundtrip() raises:
    """Train with SHUFFLED, encode/decode roundtrip works end-to-end."""
    var corpus = List[String]()
    corpus.append(String("The quick brown fox jumps over the lazy dog."))
    corpus.append(String("A completely different sentence about tokenizers."))
    corpus.append(String("Byte-level encoding must preserve all Unicode text."))

    var tok = BPETokenizer[GPT4Pretokenizer[ByteMapping.SHUFFLED]]()
    tok.train(corpus, 300)

    var text = String("The quick brown fox jumps over the lazy dog.")
    var ids = tok.encode(text)
    assert_true(len(ids) > 0, "SHUFFLED encode must produce tokens")
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


# ═══════════════════════════════════════════════════════════════
# Special token tests
# ═══════════════════════════════════════════════════════════════

def test_special_tokens_pt_mappings() raises:
    """Each PT returns the correct special token mapping."""
    var gpre = GPreTokenizer.special_tokens()
    assert_equal(len(gpre), 0)

    var gpt2 = GPT2Pretokenizer.special_tokens()
    assert_equal(gpt2["<|endoftext|>"], 50256)

    var gpt4_seq = GPT4Pretokenizer[ByteMapping.SEQUENTIAL].special_tokens()
    assert_equal(len(gpt4_seq), 5)
    assert_equal(gpt4_seq["<|endoftext|>"], 100257)
    assert_equal(gpt4_seq["<|fim_prefix|>"], 100258)
    assert_equal(gpt4_seq["<|fim_middle|>"], 100259)
    assert_equal(gpt4_seq["<|fim_suffix|>"], 100260)
    assert_equal(gpt4_seq["<|endofprompt|>"], 100276)

    var gpt4_shu = GPT4Pretokenizer[ByteMapping.SHUFFLED].special_tokens()
    assert_equal(len(gpt4_shu), 2)
    assert_equal(gpt4_shu["<|endoftext|>"], 199999)
    assert_equal(gpt4_shu["<|endofprompt|>"], 200018)


def test_special_tokens_register() raises:
    """Register special tokens and verify they're accessible."""
    var tok = BPETokenizer[GPT2Pretokenizer]()
    var specials = Dict[String, Int]()
    specials["<|endoftext|>"] = 50256
    tok.register_special_tokens(specials)
    assert_equal(len(tok.special_bytes), 1)
    assert_equal(tok.special_bytes["<|endoftext|>"], 50256)
    assert_true(50256 in tok.inverse_special)
    assert_equal(tok.inverse_special[50256], "<|endoftext|>")
    assert_true(len(tok) >= 50257)
    assert_equal(tok.vocab[50256], "<|endoftext|>")


def test_special_tokens_encode_with_special() raises:
    """Encode text containing a special token."""
    var tok = BPETokenizer[GPreTokenizer]()
    var corpus = List[String]()
    corpus.append(String("Hello world this is a test"))
    corpus.append(String("Another sentence for training"))
    tok.train(corpus, 300)

    var specials = Dict[String, Int]()
    specials["<|endoftext|>"] = 300
    tok.register_special_tokens(specials)

    var text = String("Hello <|endoftext|> world")
    var ids = tok.encode(text)
    assert_true(300 in ids, "special token ID must appear in output")

    var decoded = tok.decode(ids)
    # decode should reproduce the original text
    assert_equal(decoded, text)


def test_special_tokens_encode_without_special() raises:
    """Encode without special token — same as encode_ordinary."""
    var tok = BPETokenizer[GPreTokenizer]()
    var corpus = List[String]()
    corpus.append(String("Hello world this is a test"))
    tok.train(corpus, 300)

    var text = String("Hello world")
    var ids_special = tok.encode(text)
    var ids_ordinary = tok.encode_ordinary(text)
    assert_equal(len(ids_special), len(ids_ordinary))
    for i in range(len(ids_special)):
        assert_equal(ids_special[i], ids_ordinary[i])


def test_special_tokens_no_specials_registered() raises:
    """No specials registered — encode_ordinary is the zero-cost path."""
    var tok = BPETokenizer[GPreTokenizer]()
    var corpus = List[String]()
    corpus.append(String("Hello world this is a test"))
    tok.train(corpus, 300)
    assert_equal(len(tok.special_bytes), 0)
    var ids = tok.encode("Hello world")
    assert_true(len(ids) > 0)


def test_special_tokens_save_load() raises:
    """Special tokens survive tiktoken save/load when re-registered after load."""
    var tok = BPETokenizer[GPreTokenizer]()
    var corpus = List[String]()
    corpus.append(String("Hello world this is a test"))
    tok.train(corpus, 300)

    var specials = Dict[String, Int]()
    specials["<|endoftext|>"] = 300
    tok.register_special_tokens(specials)

    var path = "/tmp/bpe_special_save_load.tiktoken"
    tok.save_tiktoken(path)
    var loaded = BPETokenizer()
    loaded.load_tiktoken(path)

    # tiktoken format does not persist special tokens; re-register after load
    assert_equal(len(loaded.special_bytes), 0)
    loaded.register_special_tokens(specials)
    assert_equal(loaded.special_bytes["<|endoftext|>"], 300)


def test_special_tokens_tiktoken_skip() raises:
    """S(s)ave_tiktoken skips special tokens."""
    var tok = BPETokenizer[GPreTokenizer]()
    var corpus = List[String]()
    corpus.append(String("Hello world this is a test"))
    tok.train(corpus, 300)

    var specials = Dict[String, Int]()
    specials["<|endoftext|>"] = 300
    tok.register_special_tokens(specials)

    var path = "/tmp/bpe_special_skip.tiktoken"
    tok.save_tiktoken(path)

    # Load back and check specials are re-registered
    var loaded = BPETokenizer[GPreTokenizer]()
    loaded.load_tiktoken(path)
    # GPreTokenizer has no special tokens, so none should be registered
    assert_equal(len(loaded.special_bytes), 0)


def test_special_tokens_gpt2_auto_register() raises:
    """GPT2Pretokenizer auto-registers <|endoftext|> on load_tiktoken."""
    var tok = BPETokenizer[GPT2Pretokenizer]()
    var corpus = List[String]()
    corpus.append(String("Hello world this is a test"))
    tok.train(corpus, 300)
    var path = "/tmp/bpe_gpt2_special_auto.tiktoken"
    tok.save_tiktoken(path)

    var loaded = BPETokenizer[GPT2Pretokenizer]()
    loaded.load_tiktoken(path)
    assert_equal(len(loaded.special_bytes), 1)
    assert_equal(loaded.special_bytes["<|endoftext|>"], 50256)
    assert_equal(loaded.vocab[50256], "<|endoftext|>")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
