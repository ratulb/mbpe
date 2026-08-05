"""Suite 6 — deterministic fuzz parity.

Seeded stdlib ``random`` (no extra deps) generates strings from an
alphabet covering ASCII, punctuation, Latin-1, combining marks,
non-ASCII whitespace and CJK; both id-level and split-level parity are
asserted.  The seed is fixed so failures are reproducible.
"""

import random

import pytest

import parity_helpers as ph

_N_CASES = 200


def _gen_cases(seed, n=_N_CASES, min_len=1, max_len=24):
    rng = random.Random(seed)
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        yield "".join(rng.choice(ph.FUZZ_ALPHABET) for _ in range(length))


FUZZ_CASES = list(_gen_cases(0x5EED))


@pytest.mark.parametrize("text", FUZZ_CASES, ids=[f"fz{i}" for i in range(_N_CASES)])
def test_fuzz_encode_ordinary_ids(encoding_pair, text):
    tok_mbpe, tok_tk, _, _ = encoding_pair
    ph.assert_ids_match(tok_mbpe, tok_tk, text, method="encode_ordinary")


@pytest.mark.parametrize("text", FUZZ_CASES, ids=[f"fz{i}" for i in range(_N_CASES)])
def test_fuzz_split(encoding_pair, text):
    tok_mbpe, _, mbpe_name, tiktoken_name = encoding_pair
    ph.assert_split_matches(tok_mbpe, mbpe_name, tiktoken_name, text)
