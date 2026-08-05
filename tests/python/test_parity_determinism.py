"""Suite 8 — determinism parity.

Encoding must be deterministic in-process and across processes, and
training twice on the same corpus must produce byte-identical .tiktoken
files.
"""

import hashlib
import os
import subprocess
import sys
import tempfile

import pytest

import parity_helpers as ph

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT = " ".join(ph.CORPUS_PARITY)

_PT_CLASSES = {
    "gpt2": ("GPT2Tokenizer", "gpt2"),
    "gpt4": ("GPT4Tokenizer", "cl100k"),
    "gpt4o": ("GPT4oTokenizer", "o200k"),
}


class TestInProcess:
    @pytest.mark.parametrize("mbpe_name", ["gpt2", "cl100k", "o200k"])
    def test_repeat_encode_identical(self, mbpe_name):
        tok = ph.mbpe.get_encoding(mbpe_name)
        assert tok.encode_ordinary(TEXT) == tok.encode_ordinary(TEXT)

    @pytest.mark.parametrize("mbpe_name", ["gpt2", "cl100k", "o200k"])
    def test_repeat_pretokenize_identical(self, mbpe_name):
        tok = ph.mbpe.get_encoding(mbpe_name)
        assert tok.pretokenize(TEXT) == tok.pretokenize(TEXT)


class TestCrossProcess:
    def _subprocess_digest(self, mbpe_name):
        code = (
            "import sys, hashlib; sys.path.insert(0, 'python-binding');"
            "import mbpe; "
            "t = mbpe.get_encoding(%r); "
            "print(hashlib.md5(repr(t.encode_ordinary(%r)).encode()).hexdigest())"
            % (mbpe_name, TEXT)
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()

    @pytest.mark.parametrize("mbpe_name", ["gpt2", "cl100k", "o200k"])
    def test_subprocess_matches_in_process(self, mbpe_name):
        tok = ph.mbpe.get_encoding(mbpe_name)
        in_process = hashlib.md5(repr(tok.encode_ordinary(TEXT)).encode()).hexdigest()
        assert self._subprocess_digest(mbpe_name) == in_process


class TestTrainingDeterminism:
    @pytest.mark.parametrize("pt", sorted(_PT_CLASSES))
    def test_double_train_byte_identical(self, pt):
        cls_name, _ = _PT_CLASSES[pt]
        cls = getattr(ph.mbpe, cls_name)
        corpus = ph.CORPUS_PARITY + ph.CORPUS_PARITY

        t1, t2 = cls(), cls()
        t1.train(corpus, 300)
        t2.train(corpus, 300)

        paths = []
        for tok in (t1, t2):
            fd, path = tempfile.mkstemp(suffix=".tiktoken")
            os.close(fd)
            tok.save_tiktoken(path)
            paths.append(path)
        try:
            with open(paths[0], "rb") as f:
                c1 = f.read()
            with open(paths[1], "rb") as f:
                c2 = f.read()
            assert c1 == c2, "training twice produced different .tiktoken files"
        finally:
            for path in paths:
                os.unlink(path)
