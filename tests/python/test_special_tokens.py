"""Test special token registration, encoding, and save/load interaction."""

import pytest
import tempfile
import os


class TestRegister:
    def test_register_special(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        assert tok._get_registered_specials()["<|endoftext|>"] == 300

    def test_duplicate_raises(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        with pytest.raises(Exception):
            tok.register_special_tokens({"<|endoftext|>": 300})

    def test_empty_string_raises(self, empty_tok):
        with pytest.raises(Exception):
            empty_tok.register_special_tokens({"": 300})


class TestEncodeWithSpecials:
    def test_encode_with_special(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        text = "hello <|endoftext|> world"
        ids = tok.encode(text)
        assert 300 in ids
        assert tok.decode(ids) == text

    def test_encode_without_special(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        text = "hello world"
        ids = tok.encode(text)
        ids_ord = tok.encode_ordinary(text)
        assert ids == ids_ord

    def test_encode_no_specials_registered(self, trained_tok):
        tok = trained_tok
        text = "hello world"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_special_token_overlap(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"world": 300})
        text = "hello world"
        ids = tok.encode(text)
        assert 300 in ids
        assert tok.decode(ids) == text


class TestTiktokenInteraction:
    def test_special_skip_on_save(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path = f.name
        try:
            tok.save_tiktoken(path)
            with open(path) as f:
                content = f.read()
            assert "<|endoftext|>" not in content
        finally:
            os.unlink(path)

    def test_gpt2_auto_register(self):
        import mbpe
        tok = mbpe.GPT2Tokenizer()
        tok.train(["hello world"], 300)
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path = f.name
        try:
            tok.save_tiktoken(path)
            loaded = mbpe.GPT2Tokenizer()
            loaded.load_tiktoken(path)
            specials = loaded._get_registered_specials()
            assert "<|endoftext|>" in specials
        finally:
            os.unlink(path)
