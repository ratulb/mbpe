#!/usr/bin/env python3
"""Generate reference pre-tokenizer splits using Python `regex` module.

Usage:
    python benchmarks/reference_splits.py corpus.txt ref_splits.json

Output JSON structure:
    {
      "text": "...",           // the input text
      "gpre":  ["tok1", ...],  // GPreTokenizer (Ġ convention)
      "gpt2":  ["tok1", ...],  // GPT2Pretokenizer (r50k_base pattern)
      "gpt4":  ["tok1", ...],  // GPT4Pretokenizer (cl100k_base pattern)
      "n_gpre": N,             // split count
      "n_gpt2": N,
      "n_gpt4": N,
    }
"""

import json
import sys

import regex as re

# ── GPT-2 r50k_base pattern ─────────────────────────────────────
# Source: tiktoken/_tiktoken.c (and openai/gpt-2 encoder.py)
# Note: possessive quantifiers (++) prevent backtracking.
GPT2_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"

# ── GPT-4 cl100k_base pattern ────────────────────────────────────
# Source: tiktoken/_tiktoken.c
GPT4_PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"


def split_gpre(text: str) -> list[str]:
    """Replicate Mojo GPreTokenizer logic.

    1. Replace ' ' with ' Ġ' (prepend spacer before each space)
    2. Replace '.' with ' .' (separate period)
    3. Split on ' '
    """
    # Note: this matches the Mojo implementation exactly.
    spacer = "Ġ"
    t = text.replace(" ", " " + spacer)
    t = t.replace(".", " .")
    return t.split(" ")


def split_gpt2(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(GPT2_PATTERN, text)


def split_gpt4(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(GPT4_PATTERN, text)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    txt_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    gpre = split_gpre(text)
    gpt2 = split_gpt2(text)
    gpt4 = split_gpt4(text)

    out = {
        "text": text,
        "gpre": gpre,
        "gpt2": gpt2,
        "gpt4": gpt4,
        "n_gpre": len(gpre),
        "n_gpt2": len(gpt2),
        "n_gpt4": len(gpt4),
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"Wrote {out_path}", file=sys.stderr)
        print(f"  GPre bytes={len(out['gpre'])}  GPT2={len(out['gpt2'])}  GPT4={len(out['gpt4'])}", file=sys.stderr)
    else:
        json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
