"""Suite 3 — special-token parity.

The registered special-token set must match tiktoken's exactly, encode()
must insert the same single IDs, and the wrapper's
allowed_special / disallowed_special semantics must mirror tiktoken's.
"""

import pytest

import parity_helpers as ph


def _encode_catch(tok, text, **kwargs):
    try:
        return ("ok", tok.encode(text, **kwargs))
    except ValueError as e:
        return ("raise", str(e))


class TestRegisteredSpecials:
    def test_registered_set_matches_tiktoken(self, encoding_pair):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        mbpe_specials = tok_mbpe._get_registered_specials()
        tk_specials = dict(tok_tk._special_tokens)
        assert mbpe_specials == tk_specials, (
            f"mbpe={sorted(mbpe_specials.items())}\n"
            f"tikt={sorted(tk_specials.items())}"
        )

    def test_each_special_encodes_to_its_id(self, encoding_pair):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        for special, tid in tok_tk._special_tokens.items():
            assert tok_mbpe.encode(special) == [tid]
            assert tok_tk.encode(special, allowed_special="all") == [tid]

    def test_special_embedded_in_text(self, encoding_pair):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        special = list(tok_tk._special_tokens)[0]
        text = "abc" + special + "def"
        mbpe_ids = tok_mbpe.encode(text)
        tk_ids = tok_tk.encode(text, allowed_special="all")
        assert mbpe_ids == tk_ids
        assert tok_mbpe.decode(mbpe_ids) == text
        assert tok_tk.decode(tk_ids) == text


class TestDisallowedBehavior:
    def test_raise_outcome_matches_tiktoken(self, encoding_pair):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        special = list(tok_tk._special_tokens)[0]
        text = "abc" + special + "def"

        mbpe_outcome = _encode_catch(
            tok_mbpe, text, allowed_special=set(), disallowed_special="raise"
        )
        tk_outcome = _encode_catch(
            tok_tk, text, allowed_special=set(), disallowed_special="raise"
        )
        assert mbpe_outcome[0] == tk_outcome[0], (
            f"raise behavior differs: mbpe={mbpe_outcome[0]} tikt={tk_outcome[0]}"
        )
        if mbpe_outcome[0] == "ok":
            assert mbpe_outcome[1] == tk_outcome[1]

    def test_ignore_outcome_matches_tiktoken(self, encoding_pair):
        """mbpe's disallowed_special='ignore' must equal tiktoken's
        disallowed_special=() (nothing disallowed -> specials encode as
        ordinary text).  Note tiktoken has no 'ignore' string value; mbpe
        adds it as sugar for disallowed_special=()."""
        tok_mbpe, tok_tk, _, _ = encoding_pair
        special = list(tok_tk._special_tokens)[0]
        text = "abc" + special + "def"

        mbpe_ids = tok_mbpe.encode(
            text, allowed_special=set(), disallowed_special="ignore"
        )
        tk_ids = tok_tk.encode(
            text, allowed_special=set(), disallowed_special=()
        )
        assert mbpe_ids == tk_ids

    def test_subset_allowed_matches_tiktoken(self, encoding_pair):
        tok_mbpe, tok_tk, _, _ = encoding_pair
        specials = list(tok_tk._special_tokens)
        if len(specials) < 2:
            pytest.skip("need >=2 specials for a strict subset")
        allowed = {specials[0]}
        text = "x" + specials[0] + "y" + specials[1] + "z"

        mbpe_ids = tok_mbpe.encode(
            text, allowed_special=allowed, disallowed_special="ignore"
        )
        tk_ids = tok_tk.encode(
            text, allowed_special=allowed, disallowed_special=()
        )
        assert mbpe_ids == tk_ids
