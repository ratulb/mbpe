"""BPETokenizer test suite.  Run with: mojo -I . tests/test_tokenizer.mojo."""

from bpe.tokenizer import BPETokenizer
from std.testing import assert_equal, assert_true, TestSuite


def test_byte_level_no_unk() raises:
    """Verify unknown byte sequences produce 3 distinct non-zero token IDs.
    Train on "abc", encode "xyz" — since our vocab is byte-level, every
    byte (including those unseen during training) maps to a known token ID.
    No UNK token is ever produced.
    """
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
    """Train on a simple "hello world" corpus, then verify that encoding
    and decoding returns the original text byte-perfect.  This is the
    most basic sanity check for the train→encode→decode pipeline.
    """
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    assert_true(len(tok) >= 256)

    var ids = tok.encode(String("hello world"))
    assert_true(len(ids) > 0)
    assert_equal(tok.decode(ids), "hello world")


def test_empty_input() raises:
    """Edge case: empty string input for both encode and decode.
    Must return an empty list / empty string — never crash or produce
    a non-empty result.
    """
    var corpus = List[String]()
    corpus.append(String("abc"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    assert_equal(len(tok.encode(String(""))), 0)
    assert_equal(tok.decode(List[Int]()), "")


def test_deterministic() raises:
    """Verify that training twice on the same corpus yields the same
    number of merge rules.  If the algorithm is deterministic the two
    tokenizers must learn the same merge count.
    """
    var corpus = List[String]()
    corpus.append(String("hello world"))

    var tok1 = BPETokenizer()
    tok1.train(corpus, 300)
    var tok2 = BPETokenizer()
    tok2.train(corpus, 300)

    assert_equal(len(tok1.merges), len(tok2.merges))


def test_full_hf_corpus() raises:
    """Train on a realistic multi-sentence corpus (Hugging Face Course
    excerpts) and verify encode→decode roundtrip on an unseen phrase.
    Also verifies .tiktoken save/load consistency.
    """
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
    # GPT2 auto-registers <|endoftext|> (gap-padding the table), so total
    # length grows; merge rules and encode behavior are what must match.
    assert_equal(len(loaded.merges), len(tok.merges))
    assert_true(50256 in loaded.inverse_special)
    assert_equal(
        loaded.decode(loaded.encode(String("This is not a token."))),
        "This is not a token.",
    )


def test_wikipedia_example() raises:
    """Train with 3 merges on "aaabdaaabac" — the canonical BPE example
    from Wikipedia (https://en.wikipedia.org/wiki/Byte_pair_encoding).

    After 3 merges the 11-byte input should reduce to exactly 5 tokens:
      a=97, b=98, c=99, d=100 (ASCII)
      Z=aa=256, Y=ab|Za=257, X=ZY|Yb=258
    Final tokens: [258, 100, 258, 97, 99]
    """
    var corpus = List[String]()
    corpus.append(String("aaabdaaabac"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 3)

    var ids = tok.encode(String("aaabdaaabac"))
    assert_equal(len(ids), 5)
    assert_equal(ids[0], 258)
    assert_equal(ids[1], 100)
    assert_equal(ids[2], 258)
    assert_equal(ids[3], 97)
    assert_equal(ids[4], 99)
    assert_equal(tok.decode(ids), "aaabdaaabac")


def test_single_char() raises:
    """Train on repeated "aaaa" and encode the single character "a".
    A single byte should always produce exactly one token (ID 97).
    """
    var corpus = List[String]()
    corpus.append(String("aaaa"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 2)

    var ids = tok.encode(String("a"))
    assert_equal(len(ids), 1)
    assert_equal(ids[0], 97)


def test_encode_single_token() raises:
    """Verify encode_single_token on a known token, a special token,
    and that an unknown token raises.
    """
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    var token_id = tok.encode_single_token(StringSlice("hello"))
    assert_true(token_id >= 0)

    var decoded = tok.decode_single_token_bytes(token_id)
    assert_true(decoded.byte_length() > 0)

    var raised = False
    try:
        _ = tok.encode_single_token(StringSlice("\u0000nonexistent\u0000"))
    except:
        raised = True
    assert_true(raised)


def test_unicode_roundtrip() raises:
    """Verify encode→decode identity on text containing multi-byte UTF-8
    sequences (Korean Hangul, CJK, emoji).  The byte-level vocab must
    preserve arbitrary Unicode across the full pipeline.
    """
    var corpus = List[String]()
    corpus.append(String("hello world!!!? (안녕하세요!) lol123 😉"))
    var tok = BPETokenizer()
    tok.train(corpus, 256 + 32)

    var ids = tok.encode(String("hello world!!!? (안녕하세요!) lol123 😉"))
    var decoded = tok.decode(ids)
    assert_equal(decoded, "hello world!!!? (안녕하세요!) lol123 😉")


def test_token_table_copy_semantics() raises:
    """Verify TokenByteTable copy semantics: deep copy of the byte pool and
    span list.
    """
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 300)

    var table = tok.token_table
    assert_true(len(table) > 256)

    # Deep copy is independent: growing the original must not touch it.
    var before_size = len(table)
    var before_bytes = len(table.arena.bytes)
    tok.token_table.set_bytes(len(tok.token_table), String("ab").as_bytes())
    assert_true(len(tok.token_table) == before_size + 1)
    assert_true(len(table) == before_size)
    assert_true(len(table.arena.bytes) == before_bytes)

    # A second copy stays independent too (no sharing).
    var copy2 = tok.token_table
    copy2.set_bytes(len(copy2), String("cd").as_bytes())
    assert_true(len(copy2) == len(tok.token_table) + 1)
    assert_true(len(tok.token_table) == before_size + 1)
    assert_equal(tok.token_table.arena.spans[97].length, 1)

    # Tokenizer still decodes correctly after copies.
    assert_equal(tok.decode(tok.encode(String("hello world"))), "hello world")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
