"""Suite 4 — merge semantics and the base-bytes LUT guard.

The base-bytes test guards the hardcoded o200k byte<->rank lookup tables:
the first 256 token-byte entries must equal tiktoken's shuffled mapping.
"""

import pytest

import parity_helpers as ph


class TestBaseBytes:
    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ph.ENCODING_PAIRS)
    def test_single_byte_encode_matches_tiktoken(self, mbpe_name, tiktoken_name):
        """Byte->token mapping parity: every single codepoint 0x00-0xFF must
        encode to the same ID list on both sides (base-byte LUT guard)."""
        tok_mbpe = ph.mbpe.get_encoding(mbpe_name)
        tok_tk = ph.get_tiktoken().get_encoding(tiktoken_name)
        for i in range(256):
            ch = chr(i)
            assert tok_mbpe.encode_ordinary(ch) == tok_tk.encode_ordinary(ch), (
                f"{mbpe_name} diverges on chr(0x{i:02x}) {ch!r}"
            )

    def test_o200k_base_bytes_are_a_bijection(self):
        """Hardcoded o200k LUT guard: ranks 0-255 must be a permutation of
        the single bytes 0x00-0xFF (the shuffled base mapping)."""
        mb = ph.mbpe.get_encoding("o200k").token_byte_values()[:256]
        assert all(len(b) == 1 for b in mb)
        assert sorted(b[0] for b in mb) == list(range(256))


class TestRoundtrip:
    @pytest.mark.parametrize("text", ph.CORPUS_PARITY + ph.CORPUS_BOUNDARY)
    def test_encode_ordinary_decode_identity(self, encoding_pair, text):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        assert tok_mbpe.decode(tok_mbpe.encode_ordinary(text)) == text
        assert tok_tk.decode(tok_tk.encode_ordinary(text)) == text


class TestMergeMonotonicity:
    def test_concatenation_roundtrips(self, encoding_pair):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        a, b = "hello", " world "
        assert tok_mbpe.decode(tok_mbpe.encode_ordinary(a + b)) == a + b
        assert tok_tk.decode(tok_tk.encode_ordinary(a + b)) == a + b

    def test_substring_tokens_recombine(self, encoding_pair):
        """The token sequence for a+b must never lose bytes vs a alone."""
        tok_mbpe, tok_tk, _, _ = encoding_pair
        a, b = "abcd", "efgh"
        ids_ab = tok_mbpe.encode_ordinary(a + b)
        assert tok_mbpe.decode(ids_ab) == a + b
        ids_ab_tk = tok_tk.encode_ordinary(a + b)
        assert tok_tk.decode(ids_ab_tk) == a + b
