"""Test mbpe matches tiktoken on real Wikipedia content for all encodings."""

import os
import sys
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python-binding"))
import mbpe

CACHE_FILE = os.path.join(os.path.dirname(__file__), "wikipedia_unicode.txt")

WIKI_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=raw&title=Unicode"
)

_USER_AGENT = "mbpe-test/1.0"

ENCODINGS = [
    ("gpt2", "gpt2"),
    ("cl100k", "cl100k_base"),
    ("o200k", "o200k_base"),
]


def _fetch_wikipedia_text() -> str:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            text = f.read()
            if text:
                return text
    try:
        req = urllib.request.Request(WIKI_URL, headers={"User-Agent": _USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=30)
        text = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                text = f.read()
            if text:
                return text
        raise RuntimeError(f"Failed to fetch Wikipedia text and no cache: {e}") from e
    with open(CACHE_FILE, "w") as f:
        f.write(text)
    return text


@pytest.fixture(scope="session")
def wikipedia_text() -> str:
    return _fetch_wikipedia_text()


class TestWikipediaParity:
    def test_wikipedia_text_fetched(self, wikipedia_text):
        assert len(wikipedia_text) > 1000

    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_encode_matches_tiktoken(self, wikipedia_text, mbpe_name, tiktoken_name):
        import tiktoken

        tok_mbpe = mbpe.get_encoding(mbpe_name)
        tok_tiktoken = tiktoken.get_encoding(tiktoken_name)

        ids_mbpe = tok_mbpe.encode(wikipedia_text)
        ids_tiktoken = tok_tiktoken.encode(wikipedia_text)

        assert len(ids_mbpe) == len(ids_tiktoken), (
            f"Token count mismatch for {mbpe_name}: "
            f"mbpe={len(ids_mbpe)} tiktoken={len(ids_tiktoken)}"
        )

        for i, (a, b) in enumerate(zip(ids_mbpe, ids_tiktoken)):
            if a != b:
                ctx_before = wikipedia_text.encode("utf-8")[:50]
                ctx_around = wikipedia_text.encode("utf-8")[
                    max(0, i - 5) : min(len(wikipedia_text.encode("utf-8")), i + 5)
                ]
                pytest.fail(
                    f"Token ID mismatch at position {i} for {mbpe_name}: "
                    f"mbpe={a} tiktoken={b}\n"
                    f"Context around position: {ctx_around!r}"
                )

    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_decode_roundtrip(self, wikipedia_text, mbpe_name, tiktoken_name):
        tok_mbpe = mbpe.get_encoding(mbpe_name)

        ids_mbpe = tok_mbpe.encode(wikipedia_text)
        decoded = tok_mbpe.decode(ids_mbpe)

        assert decoded == wikipedia_text, (
            f"Decode roundtrip failed for {mbpe_name}: "
            f"expected {len(wikipedia_text)} chars, got {len(decoded)}"
        )

    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_tiktoken_roundtrip(self, wikipedia_text, mbpe_name, tiktoken_name):
        import tiktoken

        tok_tiktoken = tiktoken.get_encoding(tiktoken_name)

        ids = tok_tiktoken.encode(wikipedia_text)
        decoded = tok_tiktoken.decode(ids)

        assert decoded == wikipedia_text

    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_encode_ordinary_matches_encode(self, wikipedia_text, mbpe_name, tiktoken_name):
        tok_mbpe = mbpe.get_encoding(mbpe_name)

        ids_ordinary = tok_mbpe.encode_ordinary(wikipedia_text)
        ids_encode = tok_mbpe.encode(wikipedia_text)

        assert ids_ordinary == ids_encode, (
            f"encode_ordinary differs from encode for {mbpe_name}"
        )

    @pytest.mark.parametrize("mbpe_name,tiktoken_name", ENCODINGS)
    def test_encode_with_allowed_special(self, wikipedia_text, mbpe_name, tiktoken_name):
        tok_mbpe = mbpe.get_encoding(mbpe_name)
        _ = tok_mbpe.encode(wikipedia_text, allowed_special="all")
        _ = tok_mbpe.encode(wikipedia_text, allowed_special=set())
