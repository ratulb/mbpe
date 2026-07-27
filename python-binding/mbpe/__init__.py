"""Python wrapper around _mbpe (Mojo BPE tokenizer).

Adds Pythonic encode() with allowed_special/disallowed_special kwargs,
matching the tiktoken API surface. Everything else delegates to the
Mojo shared library via __getattr__.
"""
import _mbpe
import os


class _BaseTokenizer:
    """Wraps a _mbpe tokenizer adding encode() with special token params."""

    def __init__(self, *args, **kwargs):
        self._tok = self._make_tok(*args, **kwargs)

    def _get_registered_specials(self):
        return dict(self._tok._get_special_bytes())

    def encode(self, text, *, allowed_special="all", disallowed_special="raise"):
        """Encode text with special token handling.

        Parameters
        ----------
        text : str
            Input text to encode.
        allowed_special : set, "all", or None
            Set of special tokens allowed in the text. ``"all"`` (default)
            allows every registered special token.  An empty set/``None``
            means no special tokens are recognised (equivalent to
            :meth:`encode_ordinary` when combined with
            ``disallowed_special="ignore"``).
        disallowed_special : "raise" or "ignore"
            What to do when a registered special token that is *not* in
            *allowed_special* appears in *text*:

            - ``"raise"`` / ``"error"`` — raise ``ValueError``.
            - ``"ignore"`` — encode the special token as ordinary text
              (do not treat it as a single token).

        Returns
        -------
        list of int
            Token IDs.
        """
        registered = self._get_registered_specials()

        # ---- normalise allowed_special ----
        if allowed_special == "all" or allowed_special is None:
            allowed = set(registered)
        elif isinstance(allowed_special, (set, frozenset, list, tuple)):
            allowed = set(allowed_special)
        else:
            allowed = set(allowed_special) if allowed_special else set()

        for s in allowed:
            if s not in registered:
                raise ValueError(
                    f"Special token {s!r} is not registered"
                )

        # ---- normalise disallowed_special ----
        if disallowed_special not in ("raise", "error", "ignore"):
            raise ValueError(
                "disallowed_special must be 'raise', 'error', or 'ignore', "
                f"got {disallowed_special!r}"
            )

        if not registered:
            return self._tok.encode_ordinary(text)

        # ---- disallowed_special == "raise" / "error" ----
        if disallowed_special in ("raise", "error"):
            for s in registered:
                if s not in allowed and s in text:
                    raise ValueError(
                        f"Special token {s!r} is not allowed in the text"
                    )
            return self._tok.encode(text)

        # ---- disallowed_special == "ignore" ----
        if not allowed:
            return self._tok.encode_ordinary(text)

        if allowed == set(registered):
            return self._tok.encode(text)

        # Subset: manually split on allowed specials using IDs from registered
        result = []
        pos = 0
        while pos < len(text):
            earliest = len(text)
            earliest_tok = None
            earliest_id = None
            for tok in allowed:
                if tok not in registered:
                    continue
                tid = registered[tok]
                idx = text.find(tok, pos)
                if idx != -1 and idx < earliest:
                    earliest = idx
                    earliest_tok = tok
                    earliest_id = tid
            if earliest_tok is not None:
                if earliest > pos:
                    result.extend(self._tok.encode_ordinary(text[pos:earliest]))
                result.append(earliest_id)
                pos = earliest + len(earliest_tok)
            else:
                result.extend(self._tok.encode_ordinary(text[pos:]))
                break
        return result

    def load_tiktoken(self, path):
        """Load a .tiktoken file and sync special tokens."""
        self._tok.load_tiktoken(path)

    @property
    def n_vocab(self):
        return self._tok.n_vocab()

    def __getattr__(self, name):
        """Delegate everything else to the underlying _mbpe tokenizer."""
        if name in ("encode", "_tok", "_make_tok", "n_vocab",
                    "load_tiktoken", "_get_registered_specials"):
            raise AttributeError(name)
        return getattr(self._tok, name)


class GPreTokenizer(_BaseTokenizer):
    _TOK_CLS = _mbpe.GPreTokenizer
    def _make_tok(self):
        return self._TOK_CLS()


class GPT2Tokenizer(_BaseTokenizer):
    _TOK_CLS = _mbpe.GPT2Tokenizer
    def _make_tok(self):
        return self._TOK_CLS()


class GPT4Tokenizer(_BaseTokenizer):
    _TOK_CLS = _mbpe.GPT4Tokenizer
    def _make_tok(self):
        return self._TOK_CLS()


class GPT4oTokenizer(_BaseTokenizer):
    _TOK_CLS = _mbpe.GPT4oTokenizer
    def _make_tok(self):
        return self._TOK_CLS()


# ── Module-level helpers ────────────────────────────────────────

def _find_data_dir():
    mod_dir = os.path.dirname(os.path.realpath(_mbpe.__file__))
    return os.path.normpath(os.path.join(mod_dir, "..", "data"))


_ENCODING_MAP = {
    "gpt2": GPT2Tokenizer,
    "cl100k": GPT4Tokenizer,
    "o200k": GPT4oTokenizer,
}


def get_encoding(name):
    """Load a pre-built encoding from a ``.tiktoken`` file.

    Parameters
    ----------
    name : str
        One of ``"gpt2"``, ``"cl100k"``, or ``"o200k"``.

    Returns
    -------
    GPreTokenizer | GPT2Tokenizer | GPT4Tokenizer | GPT4oTokenizer
        Loaded tokenizer instance.
    """
    cls = _ENCODING_MAP.get(name)
    if cls is None:
        raise ValueError(f"unknown encoding {name!r}; use one of {list(_ENCODING_MAP)}")
    tok = cls()
    tok._tok.load_tiktoken(os.path.join(_find_data_dir(), f"{name}.tiktoken"))
    return tok


def train(texts, vocab_size):
    """Train a GPreTokenizer wrapper instance.

    Parameters
    ----------
    texts : list of str
        Training corpus.
    vocab_size : int
        Target vocabulary size (>= 256).

    Returns
    -------
    GPreTokenizer
        Trained tokenizer.
    """
    return _train_impl(texts, vocab_size, "gpre")


_PT_MAP = {
    "gpre": GPreTokenizer,
    "gpt2": GPT2Tokenizer,
    "gpt4": GPT4Tokenizer,
}


def _train_impl(texts, vocab_size, pretokenizer="gpre"):
    """Train a tokenizer with the specified pretokenizer.

    Parameters
    ----------
    texts : list of str
        Training corpus.
    vocab_size : int
        Target vocabulary size.
    pretokenizer : str
        One of ``"gpre"``, ``"gpt2"``, or ``"gpt4"``.

    Returns
    -------
    GPreTokenizer | GPT2Tokenizer | GPT4Tokenizer
        Trained tokenizer wrapper.
    """
    cls = _PT_MAP.get(pretokenizer)
    if cls is None:
        raise ValueError(f"unknown pretokenizer {pretokenizer!r}")
    tok = cls()
    tok._tok.train(texts, vocab_size)
    return tok
