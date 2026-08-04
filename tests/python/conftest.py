import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python-binding"))
import mbpe

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

TOKENIZER_CLASSES = {
    "gpt2": mbpe.GPT2Tokenizer,
    "gpt4": mbpe.GPT4Tokenizer,
    "gpt4o": mbpe.GPT4oTokenizer,
}

CORPUS = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

CANONICAL_CORPUS = ["aaabdaaabac"]

ALL_VARIANTS = ["gpt2", "gpt4", "gpt4o"]


@pytest.fixture(params=ALL_VARIANTS)
def tokenizer_type(request):
    return request.param


@pytest.fixture
def trained_tok(tokenizer_type):
    cls = TOKENIZER_CLASSES[tokenizer_type]
    tok = cls()
    tok.train(CORPUS, 300)
    return tok


@pytest.fixture
def trained_tok_canonical(tokenizer_type):
    cls = TOKENIZER_CLASSES[tokenizer_type]
    tok = cls()
    tok.train(CANONICAL_CORPUS, 259)
    return tok


@pytest.fixture
def empty_tok(tokenizer_type):
    cls = TOKENIZER_CLASSES[tokenizer_type]
    return cls()
