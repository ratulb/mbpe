"""Full-vocab decode parity: mbpe token bytes must equal tiktoken's for every
token ID in each encoding's .tiktoken file.

This is the regression gate for the o200k token-84321 corruption
(" Ġ" collapsed to "  " under the old display-string load path) and any future
load-path drift.  For each encoding we iterate every rank present in tiktoken's
mergeable ranks (the exact set of file entries) and compare
decode_single_token_bytes() on both sides.

Special token IDs that only exist in mbpe's gap-padded table (e.g. o200k's
<|endoftext|> at 199999 / <|endofprompt|> at 200018) are intentionally not
iterated here: tiktoken has no byte entry for them, while mbpe stores their raw
text bytes.  File tokens that also happen to be specials (gpt2's 50256) are
covered because they appear in mergeable ranks.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python-binding"))
import mbpe

ENCODINGS = [
    ("gpt2", "gpt2"),
    ("cl100k", "cl100k_base"),
    ("o200k", "o200k_base"),
]


class TestFullVocabDecodeParity:
    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_all_token_bytes_match_tiktoken(self, mbpe_name, tiktoken_name):
        import tiktoken

        tok_mbpe = mbpe.get_encoding(mbpe_name)
        tok_tiktoken = tiktoken.get_encoding(tiktoken_name)

        ranks = tok_tiktoken._mergeable_ranks
        assert len(ranks) > 50000

        # _mergeable_ranks maps token_bytes -> rank; iterate the rank values.
        for token_id in ranks.values():
            mbpe_bytes = tok_mbpe.decode_single_token_bytes(token_id)
            tk_bytes = tok_tiktoken.decode_single_token_bytes(token_id)
            assert mbpe_bytes == tk_bytes, (
                f"{mbpe_name} token {token_id} mismatch: "
                f"mbpe={mbpe_bytes!r} tiktoken={tk_bytes!r}"
            )

    def test_o200k_token_84321_is_gspace(self):
        """Regression gate: o200k token 84321 must decode to ' Ġ'
        (raw bytes 0x20 0xC4 0xA0), not the collapsed '  '."""
        import tiktoken

        tok_mbpe = mbpe.get_encoding("o200k")
        assert tok_mbpe.decode_single_token_bytes(84321) == b" \xc4\xa0"
        assert tok_mbpe.decode([84321]) == " Ġ"

        tok_tiktoken = tiktoken.get_encoding("o200k_base")
        assert tok_tiktoken.decode_single_token_bytes(84321) == b" \xc4\xa0"

    def test_all_encodings_n_vocab_sanity(self):
        for name in ("gpt2", "cl100k", "o200k"):
            tok = mbpe.get_encoding(name)
            assert tok.n_vocab > 50000
