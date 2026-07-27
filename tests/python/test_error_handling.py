"""Test error handling: invalid vocab sizes, out-of-range IDs, etc."""

import pytest
import os
import tempfile


class TestTrainErrors:
    def test_vocab_below_byte_range(self, empty_tok):
        with pytest.raises(Exception):
            empty_tok.train(["hello"], 255)

    def test_vocab_too_small(self, empty_tok):
        with pytest.raises(Exception):
            empty_tok.train(["hello"], 0)

    def test_vocab_negative(self, empty_tok):
        with pytest.raises(Exception):
            empty_tok.train(["hello"], -1)


class TestDecodeErrors:
    def test_decode_out_of_range(self, trained_tok):
        with pytest.raises(Exception):
            trained_tok.decode([99999])

    def test_decode_mixed_valid_invalid(self, trained_tok):
        with pytest.raises(Exception):
            trained_tok.decode([97, 99999, 98])


class TestLoadErrors:
    def test_load_nonexistent_file(self, empty_tok):
        with pytest.raises(Exception):
            empty_tok.load_tiktoken("/tmp/__nonexistent_file_x7q9.tiktoken")
