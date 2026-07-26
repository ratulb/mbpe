from tokenizer import BPETokenizer
from pretokenizer import GPreTokenizer, GPT2Pretokenizer, GPT4Pretokenizer, PreTokenizer
from std.pathlib import Path
from std.testing import assert_equal, assert_true, TestSuite

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
