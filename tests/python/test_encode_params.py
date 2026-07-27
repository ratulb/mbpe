"""Test allowed_special / disallowed_special encode parameters."""

import pytest


class TestAllowedSpecial:
    def test_default_allows_all(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        ids = tok.encode("hello <|endoftext|> world")
        assert 300 in ids

    def test_allowed_all(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        ids = tok.encode("hello <|endoftext|> world", allowed_special="all")
        assert 300 in ids

    def test_allowed_empty_set_raises(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        with pytest.raises(ValueError):
            tok.encode("hello <|endoftext|> world", allowed_special=set())

    def test_allowed_subset(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300, "<|im_start|>": 301})
        ids = tok.encode(
            "hello <|endoftext|> world",
            allowed_special={"<|endoftext|>"},
        )
        assert 300 in ids

    def test_allowed_special_not_registered_raises(self, trained_tok):
        tok = trained_tok
        with pytest.raises(ValueError):
            tok.encode("hello", allowed_special={"<|nonexistent|>"})


class TestDisallowedSpecial:
    def test_disallowed_raise_default(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        with pytest.raises(ValueError):
            tok.encode("hello <|endoftext|> world", allowed_special=set())

    def test_disallowed_raise_explicit(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        with pytest.raises(ValueError):
            tok.encode(
                "hello <|endoftext|> world",
                allowed_special=set(),
                disallowed_special="raise",
            )

    def test_disallowed_error(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        with pytest.raises(ValueError):
            tok.encode(
                "hello <|endoftext|> world",
                allowed_special=set(),
                disallowed_special="error",
            )

    def test_disallowed_ignore_empty_allowed(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300})
        ids = tok.encode(
            "hello <|endoftext|> world",
            allowed_special=set(),
            disallowed_special="ignore",
        )
        ids_ord = tok.encode_ordinary("hello <|endoftext|> world")
        assert ids == ids_ord

    def test_disallowed_ignore_subset(self, trained_tok):
        tok = trained_tok
        tok.register_special_tokens({"<|endoftext|>": 300, "<|im_start|>": 301})
        ids = tok.encode(
            "hello <|endoftext|> world",
            allowed_special={"<|endoftext|>"},
            disallowed_special="ignore",
        )
        assert 300 in ids

    def test_invalid_disallowed_raises(self, trained_tok):
        with pytest.raises(ValueError):
            trained_tok.encode("hello", disallowed_special="invalid_option")


class TestEdgeCases:
    def test_no_specials_registered(self, trained_tok):
        tok = trained_tok
        ids_default = tok.encode("hello world")
        ids_empty = tok.encode("hello world", allowed_special=set())
        ids_ignore = tok.encode(
            "hello world", allowed_special=set(), disallowed_special="ignore"
        )
        assert ids_default == ids_empty == ids_ignore

    def test_empty_text(self, empty_tok):
        assert empty_tok.encode("") == []
        assert empty_tok.encode("", allowed_special=set()) == []
        assert empty_tok.encode(
            "", allowed_special=set(), disallowed_special="ignore"
        ) == []
