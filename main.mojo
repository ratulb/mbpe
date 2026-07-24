from tokenizer import BPETokenizer
from std.testing import assert_equal, assert_true, TestSuite


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
