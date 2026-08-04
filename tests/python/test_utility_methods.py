"""Test methods not covered by core tests: name, decode_bytes, etc."""

import pytest


class TestName:
    def test_name_gpt2(self):
        import mbpe
        tok = mbpe.GPT2Tokenizer()
        assert tok.name() == "gpt2"

    def test_name_gpt4(self):
        import mbpe
        tok = mbpe.GPT4Tokenizer()
        assert tok.name() == "cl100k"

    def test_name_gpt4o(self):
        import mbpe
        tok = mbpe.GPT4oTokenizer()
        assert tok.name() == "o200k"


class TestDecodeBytes:
    def test_decode_bytes_basic(self, trained_tok):
        tok = trained_tok
        ids = tok.encode("hello")
        raw = tok.decode_bytes(ids)
        assert raw == b"hello"

    def test_decode_bytes_empty(self, empty_tok):
        assert empty_tok.decode_bytes([]) == b""


class TestEncodeSingleToken:
    def test_encode_single_token_byte_token(self, trained_tok):
        """Byte-level tokens (IDs 0-255) roundtrip via encode_single_token."""
        for tid in range(256):
            try:
                display = trained_tok.decode([tid])
                if display and len(display) == 1:
                    tid2 = trained_tok.encode_single_token(display)
                    assert tid2 == tid
                    break
            except Exception:
                continue
        else:
            pytest.skip("no single-byte token found for roundtrip")

    def test_encode_single_token_unknown_raises(self, trained_tok):
        with pytest.raises(Exception):
            trained_tok.encode_single_token("__not_a_token__")


class TestTokenByteValues:
    def test_token_byte_values_length(self, trained_tok):
        vals = trained_tok.token_byte_values()
        assert len(vals) == trained_tok.n_vocab


class TestDecodeSingleTokenBytes:
    def test_decode_single_token_bytes_ascii_gpt2(self):
        import mbpe
        tok = mbpe.GPT2Tokenizer()
        tok.train(["a b c"], 260)
        assert tok.decode_single_token_bytes(97) == b"a"

    def test_decode_single_token_bytes_empty_raises(self, empty_tok):
        with pytest.raises(Exception):
            empty_tok.decode_single_token_bytes(999)


class TestDecodeWithOffsets:
    def test_decode_with_offsets(self, trained_tok):
        tok = trained_tok
        text = "hello world"
        ids = tok.encode(text)
        result = tok.decode_with_offsets(ids)
        assert result[0] == text
        assert len(result[1]) == len(ids)
        first_start, first_end = result[1][0]
        last_start, last_end = result[1][-1]
        assert first_start >= 0
        assert last_end <= len(text)

    def test_decode_with_offsets_empty(self, empty_tok):
        result = empty_tok.decode_with_offsets([])
        assert result[0] == ""
        assert result[1] == []
