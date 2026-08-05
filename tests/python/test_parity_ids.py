"""Suite 1 — id-level encode parity.

mbpe encode_ordinary() / encode() must return the identical ID lists to
tiktoken across the parity corpus (every pretokenizer mechanism) and the
boundary corpus (strings whose word splits are known to diverge, which
changes which byte-pair merges are legal).
"""

import pytest

import parity_helpers as ph


class TestEncodeOrdinaryParity:
    @pytest.mark.parametrize("text", ph.CORPUS_PARITY)
    def test_corpus(self, encoding_pair, text):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        ph.assert_ids_match(tok_mbpe, tok_tk, text, method="encode_ordinary")

    @pytest.mark.parametrize("text", ph.CORPUS_BOUNDARY)
    def test_boundary(self, encoding_pair, text):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        ph.assert_ids_match(tok_mbpe, tok_tk, text, method="encode_ordinary")


class TestEncodeWithSpecials:
    @pytest.mark.parametrize("text", ph.CORPUS_PARITY[:16])
    def test_corpus_with_special(self, encoding_pair, text):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        special = list(tok_tk._special_tokens)[0]
        full = special + text + special
        mbpe_ids = tok_mbpe.encode(full)
        tk_ids = tok_tk.encode(full, allowed_special="all")
        assert mbpe_ids == tk_ids, (
            f"encode() with {special!r} diverges for {text[:60]!r}\n"
            f"  mbpe={mbpe_ids[:30]}\n  tikt={tk_ids[:30]}"
        )
