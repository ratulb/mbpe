"""Core functionality: train, encode, decode, roundtrip, edge cases."""

import pytest


class TestTrainAndVocab:
    def test_n_vocab(self, trained_tok):
        assert trained_tok.n_vocab >= 256

    def test_deterministic(self, trained_tok_canonical):
        tok1 = trained_tok_canonical
        from conftest import CANONICAL_CORPUS, TOKENIZER_CLASSES
        import copy
        cls = type(tok1)
        tok2 = cls()
        tok2.train(CANONICAL_CORPUS, 259)
        assert tok1.n_vocab == tok2.n_vocab

    def test_retrain_reinitializes(self, tokenizer_type):
        from conftest import CORPUS, TOKENIZER_CLASSES
        cls = TOKENIZER_CLASSES[tokenizer_type]
        tok = cls()
        tok.train(CORPUS, 300)
        v1 = tok.n_vocab
        tok.train(["different text only"], 260)
        assert tok.n_vocab == 260
        assert tok.n_vocab != v1

    def test_vocab_not_below_256(self, tokenizer_type):
        from conftest import TOKENIZER_CLASSES
        cls = TOKENIZER_CLASSES[tokenizer_type]
        tok = cls()
        with pytest.raises(Exception):
            tok.train(["hello"], 255)

    def test_no_merge_possible(self, tokenizer_type):
        from conftest import TOKENIZER_CLASSES
        cls = TOKENIZER_CLASSES[tokenizer_type]
        tok = cls()
        tok.train(["abc"], 256)
        assert tok.n_vocab == 256


class TestEncodeDecode:
    def test_basic_roundtrip(self, trained_tok):
        tok = trained_tok
        text = "Hello world this is a test"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_empty_input(self, empty_tok):
        assert empty_tok.encode("") == []
        assert empty_tok.decode([]) == ""

    def test_encode_ordinary(self, trained_tok):
        tok = trained_tok
        text = "simple text"
        ids = tok.encode(text)
        ids_ord = tok.encode_ordinary(text)
        assert ids == ids_ord

    def test_single_byte_gpt2(self):
        import mbpe
        tok = mbpe.GPT2Tokenizer()
        tok.train(["a b c"], 260)
        ids = tok.encode("a")
        assert ids == [97]

    def test_single_byte_gpt4o_shuffled(self):
        import mbpe
        tok = mbpe.GPT4oTokenizer()
        tok.train(["a b c"], 260)
        ids = tok.encode("a")
        # o200k uses SHUFFLED byte mapping, so 'a' (97) may not map to 97
        assert len(ids) == 1
        assert tok.decode(ids) == "a"

    def test_single_char_decode(self, trained_tok):
        tok = trained_tok
        ids = tok.encode("a")
        assert tok.decode(ids) == "a"

    def test_unicode_roundtrip(self, tokenizer_type):
        from conftest import TOKENIZER_CLASSES
        cls = TOKENIZER_CLASSES[tokenizer_type]
        tok = cls()
        tok.train(["Hello 世界 नमस्ते 🌍"], 300)
        text = "Hello 世界 नमस्ते 🌍"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_whitespace_only(self, tokenizer_type):
        from conftest import TOKENIZER_CLASSES
        cls = TOKENIZER_CLASSES[tokenizer_type]
        tok = cls()
        tok.train(["   \t\n  "], 260)
        text = "  \t\n "
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_punctuation_only(self, tokenizer_type):
        from conftest import TOKENIZER_CLASSES
        cls = TOKENIZER_CLASSES[tokenizer_type]
        tok = cls()
        tok.train(["!!! ??? ..."], 260)
        text = "!!! ???"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_embedded_null_byte(self, tokenizer_type):
        from conftest import TOKENIZER_CLASSES
        cls = TOKENIZER_CLASSES[tokenizer_type]
        tok = cls()
        tok.train(["a\x00b"], 260)
        text = "a\x00b"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_large_document_roundtrip(self, trained_tok):
        tok = trained_tok
        text = "The quick brown fox " * 100
        ids = tok.encode(text)
        assert tok.decode(ids) == text


class TestWikipediaExample:
    def test_canonical_bpe(self, trained_tok_canonical):
        tok = trained_tok_canonical
        ids = tok.encode("aaabdaaabac")
        assert len(ids) >= 3
        assert tok.decode(ids) == "aaabdaaabac"
