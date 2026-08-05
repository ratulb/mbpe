"""Suite 7 — serialization parity.

A .tiktoken file saved by mbpe must reconstruct a tiktoken.Encoding that
encodes exactly like the reference encoding.  This pins the file format
(base64 bytes + rank, merge ranks identical, no special-token lines).
"""

import os
import tempfile

import pytest

import parity_helpers as ph


@pytest.mark.parametrize("mbpe_name,tiktoken_name", ph.ENCODING_PAIRS)
def test_reconstructed_encoding_matches_reference(mbpe_name, tiktoken_name):
    import tiktoken as tk

    tok_mbpe = ph.mbpe.get_encoding(mbpe_name)
    ref = tk.get_encoding(tiktoken_name)

    from tiktoken.load import load_tiktoken_bpe

    fd, path = tempfile.mkstemp(suffix=".tiktoken")
    os.close(fd)
    try:
        tok_mbpe.save_tiktoken(path)
        ranks = load_tiktoken_bpe(path)

        # The saved mergeable ranks must equal the reference exactly.
        assert ranks == ref._mergeable_ranks

        reconstructed = tk.Encoding(
            name="reconstructed",
            pat_str=ref._pat_str,
            mergeable_ranks=ranks,
            special_tokens=dict(ref._special_tokens),
        )
        for text in ph.CORPUS_PARITY[:16]:
            assert tok_mbpe.encode_ordinary(text) == reconstructed.encode_ordinary(text)
            assert ref.encode_ordinary(text) == reconstructed.encode_ordinary(text)
    finally:
        os.unlink(path)


def test_saved_file_contains_no_special_lines():
    import mbpe

    tok = mbpe.get_encoding("gpt2")
    fd, path = tempfile.mkstemp(suffix=".tiktoken")
    os.close(fd)
    try:
        tok.save_tiktoken(path)
        with open(path) as f:
            content = f.read()
        assert "<|endoftext|>" not in content
    finally:
        os.unlink(path)
