"""Suite 2 — byte-level parity.

Full-vocab token-byte parity (folded in from the former
test_full_vocab_parity.py, including the o200k token-84321 regression
gate) plus encode->decode roundtrip parity on the shared corpora.
"""

import pytest

import parity_helpers as ph


class TestFullVocabDecodeParity:
    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ph.ENCODING_PAIRS)
    def test_all_token_bytes_match_tiktoken(self, mbpe_name, tiktoken_name):
        tok_mbpe = ph.mbpe.get_encoding(mbpe_name)
        tok_tk = ph.get_tiktoken().get_encoding(tiktoken_name)

        ranks = tok_tk._mergeable_ranks
        assert len(ranks) > 50000

        # One FFI call for the whole table, then Python-side compares.
        mbpe_values = tok_mbpe.token_byte_values()
        for token_id in ranks.values():
            tk_bytes = tok_tk.decode_single_token_bytes(token_id)
            assert mbpe_values[token_id] == tk_bytes, (
                f"{mbpe_name} token {token_id} mismatch: "
                f"mbpe={mbpe_values[token_id]!r} tiktoken={tk_bytes!r}"
            )

    def test_o200k_token_84321_is_gspace(self):
        """Regression gate: o200k token 84321 must decode to ' Ġ'
        (raw bytes 0x20 0xC4 0xA0), not the collapsed '  '."""
        tok_mbpe = ph.mbpe.get_encoding("o200k")
        assert tok_mbpe.decode_single_token_bytes(84321) == b" \xc4\xa0"
        assert tok_mbpe.decode([84321]) == " Ġ"

        tok_tk = ph.get_tiktoken().get_encoding("o200k_base")
        assert tok_tk.decode_single_token_bytes(84321) == b" \xc4\xa0"

    def test_all_encodings_n_vocab_sanity(self):
        for name in ("gpt2", "cl100k", "o200k"):
            assert ph.mbpe.get_encoding(name).n_vocab > 50000


class TestRoundtripParity:
    @pytest.mark.parametrize("text", ph.CORPUS_PARITY + ph.CORPUS_BOUNDARY)
    def test_decode_roundtrip(self, encoding_pair, text):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        ph.assert_bytes_match(tok_mbpe, tok_tk, text)
