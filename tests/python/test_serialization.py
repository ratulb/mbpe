"""Test .tiktoken save/load roundtrip, structure, and idempotency."""

import os
import tempfile
import pytest


class TestSaveTiktoken:
    def test_file_non_empty(self, trained_tok):
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path = f.name
        try:
            trained_tok.save_tiktoken(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) >= 256
        finally:
            os.unlink(path)

    def test_each_line_valid(self, trained_tok):
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path = f.name
        try:
            trained_tok.save_tiktoken(path)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ")
                    assert len(parts) == 2
                    assert int(parts[1]) >= 0
        finally:
            os.unlink(path)

    def test_deterministic_save(self, trained_tok):
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path2 = f.name
        try:
            trained_tok.save_tiktoken(path1)
            trained_tok.save_tiktoken(path2)
            with open(path1) as f:
                c1 = f.read()
            with open(path2) as f:
                c2 = f.read()
            assert c1 == c2
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestLoadTiktoken:
    def test_save_load_roundtrip(self, trained_tok):
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path = f.name
        try:
            trained_tok.save_tiktoken(path)
            cls = type(trained_tok)
            loaded = cls()
            loaded.load_tiktoken(path)
            text = "Hello world this is a test"
            assert loaded.decode(loaded.encode(text)) == text
        finally:
            os.unlink(path)

    def test_idempotent_save(self, trained_tok):
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path2 = f.name
        try:
            trained_tok.save_tiktoken(path1)
            cls = type(trained_tok)
            loaded = cls()
            loaded.load_tiktoken(path1)
            loaded.save_tiktoken(path2)
            with open(path1) as f:
                c1 = f.read()
            with open(path2) as f:
                c2 = f.read()
            assert c1 == c2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_no_merges_save_load_gpre(self):
        import mbpe, tempfile, os
        tok = mbpe.GPreTokenizer()
        tok.train(["abc"], 256)
        with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as f:
            path = f.name
        try:
            tok.save_tiktoken(path)
            loaded = mbpe.GPreTokenizer()
            loaded.load_tiktoken(path)
            assert loaded.n_vocab == 256
            assert loaded.encode("abc") == [97, 98, 99]
        finally:
            os.unlink(path)
