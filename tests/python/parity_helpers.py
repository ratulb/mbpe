"""Shared fixtures/assertions for the mbpe <-> tiktoken parity suites.

Every parity suite runs over the same three encoding pairs.  Names follow
the established convention: the mbpe side uses its ``get_encoding`` names
(gpt2/cl100k/o200k), the tiktoken side the real ones
(gpt2/cl100k_base/o200k_base).
"""

import pytest
import regex as _re

import mbpe

ENCODING_PAIRS = [
    ("gpt2", "gpt2"),
    ("cl100k", "cl100k_base"),
    ("o200k", "o200k_base"),
]

# Sentences exercising every pretokenizer mechanism: ASCII words, space
# runs, contractions, digits, punctuation, newlines, Unicode letters,
# combining marks, non-ASCII whitespace, CJK.
CORPUS_PARITY = [
    "Hello world",
    "  double   space runs  ",
    "don't stop believin'",
    "nums: 12, 34.5 and \uff16\uff17\uff18",
    "newline\nthen\r\nCRLF\r\n\n",
    "e\u0301 with combining accent",
    "no-break\u00a0space\u00a0run",
    "ideographic\u3000space",
    "Devanagari: \u0928\u092e\u0938\u094d\u0924\u0947",
    "Tamil: \u0bb5\u0bbe, \u0ba3",
    "CJK \u3002\u3001\u300c\u300d brackets",
    "mixed \u00c0\u00c1\u00c2 cased \u03b1\u03b2 Greek",
    "tokens\u1f00 archaic\u1f01 greek",
    "W\u00c9\u00c9 accented caps",
    "emoji \U0001f600\u2728 \U0001f4a9",
    "tabular\tand vertical\vspace",
    "U+2028 line\u2028sep U+2029 para\u2029sep",
    "CRLF split \r\n\r\n between",
    "latin \u00e9\u00e8\u00ea and greek \u03b1\u03b2\u03b3",
    "fullwidth digits \uff11\uff12\uff13 and letters \uff21\uff42",
    "arabic: \u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645",
    "hebrew: \u05e9\u05dc\u05d5\u05dd",
    "cyrillic: \u041f\u0440\u0438\u0432\u0435\u0442",
    "armenian: \u0532\u0561\u0580\u0587",
    "georgian: \u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0",
    "thai: \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
    "korean: \ud55c\uae00 \uac01",
    "japanese: \u65e5\u672c\u8a9e \u3042\u3044\u3046",
    "chinese: \u4f60\u597d\u4e16\u754c",
    "kannada: \u0c95\u0ca8\u0ccd\u0ca8\u0ca1",
    "malayalam: \u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02",
    "telugu: \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41",
    "gurmukhi: \u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40",
    "gujarati: \u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0",
    "oriya: \u0b13\u0b21\u0b3f\u0b06",
    "myanmar: \u1019\u103c\u1014\u103a\u1019\u102c",
    "khmer: \u1797\u17b6\u179f\u17b6\u1781\u17d2\u179a",
    "lao: \u0eaa\u0eb0\u0e9a\u0eb2\u0e8d\u0eb2",
    "tibetan: \u0f56\u0f7c\u0f51\u0f0b\u0f61\u0f72\u0f42",
    "mongolian: \u1218\u12a8\u122d\u130c\u12ae",
    "ethiopic: \u1205\u120d\u1208",
    "cherokee: \u13a1\u13a2\u13d2\u13d3",
]

# Deterministic byte-level fuzz alphabet: ASCII letters/digits/space,
# punctuation, Latin-1, combining marks, non-ASCII whitespace, CJK.
FUZZ_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
    ".,;:!?\"'()[]{}<>/\\|`~@#$%^&*+-_="
    "\n\r\t"
    "\u00c0\u00e0\u00a0\u00e9\u0301"
    "\u3000\u3002\u3001\u4e60\u65e5"
    "\u0928\u094d\u0bb5\u0bbe\u03b1\u1f00"
)

# Short strings whose splits diverge from tiktoken (per the empirical
# pretokenizer study) and where the differing word boundaries can change
# which byte-pair merges are legal -> id-level divergence.
CORPUS_BOUNDARY = [
    "\u3000\u3000x",          # U+3000 mis-classified as a letter (Gap B/D)
    "\u3000x\u3000\u3000",    # same, trailing
    "\u00a0\u00a0x\u00a0",    # NBSP run + trailing (Gap D)
    "ab\u00c0c",              # o200k cased-uppercase mid-run (Gap E)
    "\u00c0\u00c0\u00c0\u00c0c",  # long upper run in o200k
    "namaste\u0928\u092e\u0938\u094d\u0924\u0947",  # marks (Gap A)
    "va,\u0bb5\u0bbe,\u0ba3",  # Tamil unclassified (Gap C)
    "a\u3002b",              # CJK punct treated as letter (Gap B)
    "\u4e60\u3001\u4e61",    # CJK punct between letters (Gap B)
    "e\u0301x\u0301",        # combining marks (Gap A)
]


def get_tiktoken():
    import tiktoken

    return tiktoken


def _tiktoken_pat(tiktoken_name):
    """Cache tiktoken's compiled pretokenizer regex (the split reference)."""
    cache = globals().setdefault("_PAT_CACHE", {})
    if tiktoken_name not in cache:
        cache[tiktoken_name] = _re.compile(
            get_tiktoken().get_encoding(tiktoken_name)._pat_str
        )
    return cache[tiktoken_name]


def reference_splits(tiktoken_name, text):
    """tiktoken's word boundaries as a list of bytes (regex module proxy
    for the Rust regex crate)."""
    return [m.group().encode("utf-8") for m in _tiktoken_pat(tiktoken_name).finditer(text)]


def first_divergence(seq_a, seq_b):
    """Return (index, a, b) of the first differing element, else (None, None, None)."""
    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            return i, a, b
    if len(seq_a) != len(seq_b):
        return min(len(seq_a), len(seq_b)), "<len>", (len(seq_a), len(seq_b))
    return None, None, None


def _ctx(seq, i, width=6):
    return seq[max(0, i - width): i + width + 1]


def assert_ids_match(tok_mbpe, tok_tk, text, method="encode"):
    """Encode `text` on both sides and require identical ID lists.

    On divergence, re-encodes both sides with encode_ordinary() to
    attribute the failure: if ordinary encodings agree, the bug is in
    the Python wrapper's special-token handling; otherwise it is in the
    Mojo core (merge table, pretokenizer, byte mapping).
    """
    mbpe_ids = getattr(tok_mbpe, method)(text)
    tk_ids = getattr(tok_tk, method)(text)
    i, a, b = first_divergence(mbpe_ids, tk_ids)
    if i is None:
        return

    mbpe_ord = tok_mbpe.encode_ordinary(text)
    tk_ord = tok_tk.encode_ordinary(text)
    ord_agree = mbpe_ord == tk_ord
    blame = "wrapper special-token handling" if ord_agree else "Mojo core (merge/pretokenize/byte-map)"

    raise AssertionError(
        f"{method}() diverges at id[{i}]: mbpe={a!r} tiktoken={b!r}\n"
        f"  mbpe ctx={_ctx(mbpe_ids, i)!r}\n"
        f"  tikt ctx={_ctx(tk_ids, i)!r}\n"
        f"  len: mbpe={len(mbpe_ids)} tiktoken={len(tk_ids)}\n"
        f"  ordinary ids agree: {ord_agree} -> {blame}\n"
        f"  text={text[:120]!r}"
    )


def assert_bytes_match(tok_mbpe, tok_tk, text):
    """Encode both sides and require identical decode()-produced strings."""
    mbpe_out = tok_mbpe.decode(tok_mbpe.encode_ordinary(text))
    tk_out = tok_tk.decode(tok_tk.encode_ordinary(text))
    assert mbpe_out == tk_out, (
        f"roundtrip mismatch:\n  mbpe={mbpe_out!r}\n  tk={tk_out!r}\n  text={text[:120]!r}"
    )


def assert_split_matches(tok_mbpe, mbpe_name, tiktoken_name, text):
    """mbpe's pretokenize() word views must match tiktoken's regex split."""
    mbpe_words = list(tok_mbpe.pretokenize(text))
    tk_words = reference_splits(tiktoken_name, text)
    i, a, b = first_divergence(mbpe_words, tk_words)
    assert i is None, (
        f"{mbpe_name} split diverges at word[{i}]:\n"
        f"  mbpe={a!r}\n  tikt={b!r}\n"
        f"  mbpe ctx={_ctx(mbpe_words, i)!r}\n"
        f"  tikt ctx={_ctx(tk_words, i)!r}\n"
        f"  text={text[:120]!r}"
    )
