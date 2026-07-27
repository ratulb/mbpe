# mbpe

BPE tokenizer — Mojo implementation with Python bindings.

```python
import mbpe
tok = mbpe.get_encoding("gpt2")
tok.encode("hello world")
```

Supports gpt2, cl100k, o200k encodings plus custom training via GPreTokenizer, GPT2Tokenizer, GPT4Tokenizer.
