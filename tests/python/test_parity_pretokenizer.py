"""Suite 5 — split-level parity.

mbpe's pretokenize() word views must match tiktoken's regex split
exactly.  This is the most sensitive suite: it pins the byte offsets of
every word boundary, so it is the primary driver for the pretokenizer
gap fixes (marks, CJK punctuation/space, script coverage, Unicode
whitespace, o200k case splitting).
"""

import pytest

import parity_helpers as ph


class TestSplitParity:
    @pytest.mark.parametrize("text", ph.CORPUS_PARITY)
    def test_corpus(self, encoding_pair, text):
        tok_mbpe, _, mbpe_name, tiktoken_name = encoding_pair
        ph.assert_split_matches(tok_mbpe, mbpe_name, tiktoken_name, text)

    @pytest.mark.parametrize("text", ph.CORPUS_BOUNDARY)
    def test_boundary(self, encoding_pair, text):
        tok_mbpe, _, mbpe_name, tiktoken_name = encoding_pair
        ph.assert_split_matches(tok_mbpe, mbpe_name, tiktoken_name, text)

    def test_empty_string(self, encoding_pair):
        tok_mbpe, _, _, _ = encoding_pair
        assert tok_mbpe.pretokenize("") == []

    def test_single_codepoint(self, encoding_pair):
        tok_mbpe, _, _, _ = encoding_pair
        for ch in ("a", " ", "\u4e60", "\u00a0"):
            assert b"".join(tok_mbpe.pretokenize(ch)) == ch.encode("utf-8")

    def test_words_reassemble_to_input_bytes(self, encoding_pair):
        """The split must be a lossless partition of the input bytes."""
        tok_mbpe, _, _, _ = encoding_pair
        text = " ".join(ph.CORPUS_PARITY)
        assert b"".join(tok_mbpe.pretokenize(text)) == text.encode("utf-8")
