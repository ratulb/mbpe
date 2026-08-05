"""Test mbpe matches tiktoken on single words long enough (>= SCAN_LIMIT = 32
bytes) to take the heap-driven merge path in encode_ordinary().

The rest of the suite's natural-language inputs rarely produce a single
pre-tokenized word of >= 32 bytes, so this test exists to guarantee the
O(N log N) heap path is exercised and byte-exact.
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

# SCAN_LIMIT = 32 bytes: any single word at least this long takes the
# heap-driven merge. Pure letter runs so each pre-tokenizer keeps the word
# whole. The repeated-letter run merges heavily on all three vocabs.
LONG_WORDS = [
    "supercalifragilisticexpialidocious",  # 34 bytes
    "pneumonoultramicroscopicsilicovolcanoconiosis",  # 45 bytes
    "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqqrrrsssttt",  # 60 bytes
]

assert all(len(word) >= 32 for word in LONG_WORDS)


class TestLongWordParity:
    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_long_words_match_tiktoken(self, mbpe_name, tiktoken_name):
        import tiktoken

        tok_mbpe = mbpe.get_encoding(mbpe_name)
        tok_tiktoken = tiktoken.get_encoding(tiktoken_name)

        for word in LONG_WORDS:
            ids_mbpe = tok_mbpe.encode(word)
            ids_tiktoken = tok_tiktoken.encode(word)
            assert ids_mbpe == ids_tiktoken, (
                f"{mbpe_name} long-word mismatch for {word!r}: "
                f"mbpe={ids_mbpe} tiktoken={ids_tiktoken}"
            )

    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_long_words_decode_roundtrip(self, mbpe_name, tiktoken_name):
        tok_mbpe = mbpe.get_encoding(mbpe_name)
        for word in LONG_WORDS:
            ids = tok_mbpe.encode(word)
            assert tok_mbpe.decode(ids) == word, (
                f"{mbpe_name} long-word decode roundtrip failed for {word!r}"
            )
