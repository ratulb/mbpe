"""Exhaustive BPETokenizer test suite — Phase 1: 20 safe tests.

Run: mojo -I . tests/exhaustive_tokenizer.mojo

These tests require no source changes.  They cover:
  - Vocabulary integrity (unique IDs, correct size)
  - Training on diverse inputs (emojis, mixed scripts, repeated chars/words)
  - Merge correctness (most-freq-first, tie-breaking, grow rate)
  - Termination with oversized vocab requests
  - Encode/decode edge cases (single byte, whitespace, large docs, stability)
  - Property-based invariants (merge ranks increasing)
"""

from bpe.tokenizer import BPETokenizer
from std.testing import assert_equal, assert_true, TestSuite


# ═════════════════════════════════════════════════════════════════════════════
# Section 1 — VOCABULARY & TOKEN MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def test_vocab_unique_ids() raises:
    """Verify no two vocab entries have identical raw bytes (bytes are a
    bijection now that GPre's collapse is gone)."""
    var corpus = List[String]()
    corpus.append(String("The quick brown fox jumps over the lazy dog."))
    corpus.append(String("A completely different sentence about tokenizers."))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var values = tok.token_byte_values()
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert_true(values[i] != values[j])


def test_vocab_size_matches_expected() raises:
    """After training len(vocab) == 256 + len(merges)."""
    var corpus = List[String]()
    corpus.append(String("hello world the quick brown fox tokenizer test corpus"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    assert_true(len(tok) >= 256)
    assert_equal(len(tok), 256 + len(tok.merges))


# ═════════════════════════════════════════════════════════════════════════════
# Section 3 — MERGE RULES
# ═════════════════════════════════════════════════════════════════════════════

def test_no_merge_possible_returns_input_unchanged() raises:
    """Train with base vocab only (vocab_size=256), encode unseen text.
    Every byte stays as its own token ID — no merges applied."""
    var corpus = List[String]()
    corpus.append(String("xyz"))
    var tok = BPETokenizer()
    tok.train(corpus, 256)
    assert_equal(len(tok.merges), 0)
    var ids = tok.encode(String("abc"))
    assert_equal(len(ids), 3)
    assert_equal(ids[0], 97)
    assert_equal(ids[1], 98)
    assert_equal(ids[2], 99)
    assert_equal(tok.decode(ids), "abc")


# ═════════════════════════════════════════════════════════════════════════════
# Section 4 — TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def test_train_on_emojis() raises:
    """Train and roundtrip text containing emoji sequences."""
    var corpus = List[String]()
    corpus.append(String("😀🎉✨🚀🌟💫🎊🎈"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 8)
    var text = String("😀🎉")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


def test_train_on_mixed_languages() raises:
    """Train and roundtrip text mixing Latin, CJK, Greek, Arabic."""
    var corpus = List[String]()
    corpus.append(String("Hello World 你好 こんにちは γειά σου مرحبا"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 16)
    var text = String("Hello 你好 γειά")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)

def test_train_on_long_text_sparsed_cjk_language_tokens() raises:
    """Train and roundtrip text mixing CJK."""

    var long_text = """
    CJK stands for Chinese, Japanese, and Korean. It's a term used in computing and typography to refer to the shared writing systems of these three East Asian languages.

    Key Points about CJK:
    Shared Characters: All three languages use (or historically used) Chinese characters (Hanzi in Chinese, Kanji in Japanese, Hanja in Korean).

    Unicode CJK Unified Ideographs: In Unicode, many of these shared characters are "unified" — meaning the same underlying code point is used for characters that are essentially the same across these languages, even if the visual appearance varies slightly between them.

    Examples:

    水 (water) — used in all three languages

    中 (middle/China) — used in all three

    人 (person) — used in all three

    Sometimes CJKV: The "V" stands for Vietnamese, as Vietnamese also historically used Chinese characters (Chữ Nôm).

    Why CJK is Important:
    Text Processing: These languages don't use spaces between words and have thousands of characters, making text processing different from Latin-based languages.

    Encoding: Historically, CJK characters required special handling in character encodings (like Shift-JIS, Big5, etc.) before Unicode became standard.

    Fonts: CJK fonts need to support thousands of characters, making them much larger than Latin fonts.
    """

    var corpus = [long_text^]
    var tok = BPETokenizer()
    tok.train(corpus, 500)
    var text = String("水(water), 中 (middle/China), 人 (person)")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)



def test_train_on_single_character_repeated() raises:
    """Train on repeated single character — verifies degenerate frequency table."""
    var corpus = List[String]()
    corpus.append(String("aaaaaa"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 4)
    var text = String("aaa")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


def test_train_on_single_word_repeated() raises:
    """Train on repeated word — verifies merge learning on bounded pattern."""
    var corpus = List[String]()
    corpus.append(String("hellohellohello"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 4)
    var text = String("hello")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


def test_train_vocab_size_larger_than_achievable_terminates() raises:
    """Requesting far more merges than the corpus can produce must terminate
    cleanly (no infinite loop)."""
    var corpus = List[String]()
    corpus.append(String("ab"))
    var tok = BPETokenizer()
    tok.train(corpus, 1000)
    assert_true(len(tok) >= 256)
    assert_true(len(tok.merges) >= 1)
    # The training loop exited because max_freq <= 0, not because
    # vocab_size was reached.  This is the termination guarantee.
    assert_true(len(tok) < 1000)


def test_train_tie_breaking_is_deterministic() raises:
    """Equal-frequency pairs result in identical merge lists across runs."""
    var corpus = List[String]()
    corpus.append(String("ab cd"))
    var tok1 = BPETokenizer()
    tok1.train(corpus, 256 + 3)
    var tok2 = BPETokenizer()
    tok2.train(corpus, 256 + 3)
    assert_equal(len(tok1.merges), len(tok2.merges))
    for i in range(len(tok1.merges)):
        assert_equal(tok1.merges[i].first, tok2.merges[i].first)
        assert_equal(tok1.merges[i].second, tok2.merges[i].second)
        assert_equal(tok1.merges[i].merged, tok2.merges[i].merged)


def test_train_most_frequent_pair_merges_first() raises:
    """In a synthetic corpus where one pair clearly dominates, it must be
    the first merge rule."""
    var corpus = List[String]()
    corpus.append(String("ababab"))  # (97,98) appears 3 times — no other pair appears more than once
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 2)
    assert_true(len(tok.merges) >= 1)
    assert_equal(tok.merges[0].first, 97)
    assert_equal(tok.merges[0].second, 98)
    assert_equal(tok.merges[0].merged, 256)


def test_train_vocab_grows_one_token_per_merge() raises:
    """Every merge step adds exactly one token to the vocabulary."""
    var corpus = List[String]()
    corpus.append(String("hello world the quick brown fox tokenizer"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    assert_equal(len(tok) - 256, len(tok.merges))


def test_train_does_not_mutate_input_corpus() raises:
    """Training must not modify the caller's corpus Span."""
    var text1 = String("hello world")
    var text2 = String("foo bar baz")
    var original1 = text1.copy()
    var original2 = text2.copy()
    var corpus = List[String]()
    corpus.append(text1)
    corpus.append(text2)
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    assert_equal(text1, original1)
    assert_equal(text2, original2)


def test_retrain_reinitializes_state() raises:
    """Training a second time must reset state, not append."""
    var corpus1 = List[String]()
    corpus1.append(String("hello world"))
    var corpus2 = List[String]()
    corpus2.append(String("totally different corpus"))
    var tok = BPETokenizer()
    tok.train(corpus1, 256 + 10)
    tok.train(corpus2, 256 + 10)
    # After retraining, the merges should be from corpus2, not corpus1
    assert_true(len(tok.merges) > 0)
    # Verify encode of corpus2 text works
    var ids = tok.encode(String("totally different corpus"))
    var decoded = tok.decode(ids)
    assert_equal(decoded, "totally different corpus")


# ═════════════════════════════════════════════════════════════════════════════
# Section 6 — ENCODING
# ═════════════════════════════════════════════════════════════════════════════

def test_encode_single_byte() raises:
    """Encoding a single ASCII byte returns that byte's token ID."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var ids = tok.encode(String("a"))
    assert_equal(len(ids), 1)
    assert_equal(ids[0], 97)
    assert_equal(tok.decode(ids), "a")


def test_encode_decode_whitespace() raises:
    """Tab, newline, and multiple spaces roundtrip correctly."""
    var corpus = List[String]()
    corpus.append(String("hello\tworld\nfoo  bar"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 10)
    var text = String("hello\tworld\nfoo  bar")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


# ═════════════════════════════════════════════════════════════════════════════
# Section 7 — DECODING
# ═════════════════════════════════════════════════════════════════════════════

def test_decode_single_token() raises:
    """Decoding a single byte-level token ID returns the expected character."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var ids97 = List[Int]()
    ids97.append(97)
    var decoded = tok.decode(ids97)
    assert_equal(decoded, "a")
    var ids32 = List[Int]()
    ids32.append(32)
    decoded = tok.decode(ids32)
    assert_equal(decoded, " ")


# ═════════════════════════════════════════════════════════════════════════════
# Section 8 — ENCODE/DECODE ROUND-TRIP
# ═════════════════════════════════════════════════════════════════════════════

def test_encode_decode_roundtrip_large_document() raises:
    """A multi-kilobyte document round-trips exactly."""
    var paragraph = String(
        "The quick brown fox jumps over the lazy dog. "
        "This pangram contains every letter of the English alphabet "
        "at least once.  Tokenizers must handle long documents without "
        "introducing errors or crashing.  Byte pair encoding learns "
        "subword units by iteratively merging the most frequent adjacent "
        "byte pairs in the training corpus.  This process continues until "
        "a desired vocabulary size is reached or no more merges are possible.  "
        "The result is a set of merge rules that can encode any text, "
        "including words unseen during training, by decomposing them into "
        "known subword tokens."
    )
    var corpus = List[String]()
    corpus.append(paragraph)
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 50)
    var ids = tok.encode(paragraph)
    var decoded = tok.decode(ids)
    assert_equal(decoded, paragraph)


def test_encode_decode_roundtrip_stable_across_repetitions() raises:
    """Encode-decode chain converges to a fixed point (no drift)."""
    var corpus = List[String]()
    corpus.append(String("hello world this is a stability test"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var text = String("hello world stability test")
    var ids1 = tok.encode(text)
    var text2 = tok.decode(ids1)
    var ids2 = tok.encode(text2)
    assert_equal(len(ids2), len(ids1))
    for i in range(len(ids1)):
        assert_equal(ids2[i], ids1[i])


def test_encode_decode_roundtrip_emojis() raises:
    """Emoji text round-trips exactly (combines 6.6 + 8.3)."""
    var corpus = List[String]()
    corpus.append(String("😀🎉✨🚀🌟💫🎊🎈👋🌍🔥🌈⭐"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 16)
    var text = String("🌟🚀🌈😀")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


# ═════════════════════════════════════════════════════════════════════════════
# Section 13 — PROPERTY-BASED TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_property_merge_ranks_strictly_increasing() raises:
    """Merge rule ranks (merged token IDs) are strictly increasing
    in training order: first merge → ID 256, second → 257, etc."""
    var corpus = List[String]()
    corpus.append(String("hello world the quick brown fox jumps over"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    for i in range(len(tok.merges)):
        assert_equal(tok.merges[i].merged, 256 + i)


# ═════════════════════════════════════════════════════════════════════════════
# Section 0 — INITIALIZATION & CONFIGURATION (validation)
# ═════════════════════════════════════════════════════════════════════════════

def test_vocab_size_below_byte_range_raises_valueerror() raises:
    """V(v)ocab_size < 256 must raise — can't hold base byte vocabulary."""
    var corpus = List[String]()
    corpus.append(String("hello"))
    var raised = False
    try:
        var tok = BPETokenizer()
        tok.train(corpus, 100)
    except:
        raised = True
    assert_true(raised)


# ═════════════════════════════════════════════════════════════════════════════
# Section 4 — TRAINING (validation)
# ═════════════════════════════════════════════════════════════════════════════

def test_train_vocab_size_smaller_than_initial_raises_valueerror() raises:
    """V(v)ocab_size < base alphabet (256) raises before any training work."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var raised = False
    try:
        var tok = BPETokenizer()
        tok.train(corpus, 200)
    except:
        raised = True
    assert_true(raised)


def test_train_zero_or_negative_vocab_size_raises_valueerror() raises:
    """Zero or negative vocab_size raises before any training work."""
    var corpus = List[String]()
    corpus.append(String("hello"))
    # Test vocab_size = 0
    var raised = False
    try:
        var tok = BPETokenizer()
        tok.train(corpus, 0)
    except:
        raised = True
    assert_true(raised)
    # Test vocab_size = -1
    raised = False
    try:
        var tok = BPETokenizer()
        tok.train(corpus, -1)
    except:
        raised = True
    assert_true(raised)


# ═════════════════════════════════════════════════════════════════════════════
# Section 7 — DECODING (validation)
# ═════════════════════════════════════════════════════════════════════════════

def test_decode_out_of_range_token_id_raises_indexerror() raises:
    """Token ID beyond valid vocab range must raise."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var ids = List[Int]()
    ids.append(len(tok) + 100)
    var raised = False
    try:
        var _ = tok.decode(ids)
    except:
        raised = True
    assert_true(raised)


def test_decode_mixed_valid_invalid_ids_raises_indexerror() raises:
    """Sequence mixing valid and out-of-range IDs must raise."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var ids = List[Int]()
    ids.append(97)
    ids.append(99999)
    ids.append(98)
    var raised = False
    try:
        var _ = tok.decode(ids)
    except:
        raised = True
    assert_true(raised)


# ═════════════════════════════════════════════════════════════════════════════
# Section 9 — SPECIAL TOKENS (validation)
# ═════════════════════════════════════════════════════════════════════════════

def test_special_token_empty_string_raises_valueerror() raises:
    """Registering an empty string as a special token must raise."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var specials = Dict[String, Int]()
    specials[""] = 300
    var raised = False
    try:
        tok.register_special_tokens(specials)
    except:
        raised = True
    assert_true(raised)


def test_special_tokens_duplicate_raises_valueerror() raises:
    """Registering the same special token string twice must raise."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var specials = Dict[String, Int]()
    specials["<|endoftext|>"] = 300
    tok.register_special_tokens(specials)
    var raised = False
    try:
        tok.register_special_tokens(specials)
    except:
        raised = True
    assert_true(raised)


# ═════════════════════════════════════════════════════════════════════════════
# Section 9 — SPECIAL TOKENS (overlap)
# ═════════════════════════════════════════════════════════════════════════════

def test_special_token_overlap_with_surrounding_text() raises:
    """A special token embedded in ordinary text is preserved as a single ID."""
    var corpus = List[String]()
    corpus.append(String("hello world this is a test"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var specials = Dict[String, Int]()
    specials["<|a|>"] = 300
    tok.register_special_tokens(specials)
    var text = String("hello<|a|>world")
    var ids = tok.encode(text)
    assert_true(300 in ids)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


# ═════════════════════════════════════════════════════════════════════════════
# Section 10 — BYTE / UNICODE EDGE CASES
# ═════════════════════════════════════════════════════════════════════════════

def test_handles_embedded_null_byte() raises:
    """String containing embedded \\0 encodes and decodes correctly."""
    var corpus = List[String]()
    corpus.append(String("hello world null test"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var bytes = List[UInt8]()
    bytes.append(UInt8(0x68))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x69))
    var text = String(from_utf8_lossy=Span[UInt8](bytes))
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


def test_whitespace_only_input() raises:
    """Input containing only spaces, tabs, and newlines round-trips cleanly."""
    var corpus = List[String]()
    corpus.append(String("hello world foo bar baz"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var text = String("   \t\n  \t")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


def test_punctuation_only_input() raises:
    """Input containing only punctuation characters round-trips cleanly."""
    var corpus = List[String]()
    corpus.append(String("hello world test sentence"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var text = String("!@#$%^&*()_+-=[]{}|;:',.<>?/`~")
    var ids = tok.encode(text)
    var decoded = tok.decode(ids)
    assert_equal(decoded, text)


def test_very_long_unsplittable_token() raises:
    """A very long string with no natural split point encodes and decodes
    without error (no stack overflow or quadratic blowup death)."""
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var sb = List[String](capacity=40)
    for _ in range(40):
        sb.append(String("abcdefghijklmnopqrstuvwxyz0123456789"))
    var long_word = String()
    for s in sb:
        long_word += s
    var ids = tok.encode(long_word)
    var decoded = tok.decode(ids)
    assert_equal(decoded, long_word)


# ═════════════════════════════════════════════════════════════════════════════
# Section 13 — PROPERTY-BASED TESTS (continued)
# ═════════════════════════════════════════════════════════════════════════════

def test_property_every_token_is_decodable() raises:
    """Every token ID in the vocabulary can be decoded without raising."""
    var corpus = List[String]()
    corpus.append(String("The quick brown fox jumps over the lazy dog."))
    corpus.append(String("Tokenizers must handle various text inputs."))
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    for id in range(len(tok)):
        var ids = List[Int]()
        ids.append(id)
        var _ = tok.decode(ids)  # Must not raise


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
