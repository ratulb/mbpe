"""Test module-level functions: get_encoding, train, _train_impl."""

import pytest
import os


class TestGetEncoding:
    def _check_tiktoken_file(self, name):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        path = os.path.join(data_dir, f"{name}.tiktoken")
        if not os.path.exists(path):
            pytest.skip(f"{name}.tiktoken not found")
        return path

    def test_get_encoding_gpt2(self):
        import mbpe
        self._check_tiktoken_file("gpt2")
        tok = mbpe.get_encoding("gpt2")
        assert tok.name() == "gpt2"
        assert tok.n_vocab > 50000
        text = "Hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_get_encoding_cl100k(self):
        import mbpe
        self._check_tiktoken_file("cl100k")
        tok = mbpe.get_encoding("cl100k")
        assert tok.name() == "cl100k"
        assert tok.n_vocab > 100000
        text = "Hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_get_encoding_o200k(self):
        import mbpe
        self._check_tiktoken_file("o200k")
        tok = mbpe.get_encoding("o200k")
        assert tok.name() == "o200k"
        assert tok.n_vocab > 199000
        text = "Hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_get_encoding_unknown(self):
        import mbpe
        with pytest.raises(ValueError):
            mbpe.get_encoding("unknown")


class TestModuleTrain:
    def test_train_basic(self):
        import mbpe
        tok = mbpe.train(["hello world"], 256)
        assert tok.name() == "gpre"
        assert tok.n_vocab == 256

    def test_train_impl_gpt2(self):
        import mbpe
        tok = mbpe._train_impl(["hello world"], 256, "gpt2")
        assert tok.name() == "gpt2"
        assert tok.n_vocab == 256

    def test_train_impl_gpt4(self):
        import mbpe
        tok = mbpe._train_impl(["hello world"], 256, "gpt4")
        assert tok.name() == "cl100k"
        assert tok.n_vocab == 256

    def test_train_impl_invalid(self):
        import mbpe
        with pytest.raises(ValueError):
            mbpe._train_impl(["hello world"], 300, "invalid")
