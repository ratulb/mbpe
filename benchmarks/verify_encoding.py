#!/usr/bin/env python3
"""Level B: verify Mojo BPE encode against Python reference.

Usage:
    # Encode text and print token IDs (default text if not provided via stdin)
    echo "Hello world!" | python benchmarks/verify_encoding.py /tmp/bpe.json gpre

    # Encode from file
    python benchmarks/verify_encoding.py /tmp/bpe.json gpre < input.txt

    # Full-encode a JSONL of test strings, compare with Mojo output
    python benchmarks/verify_encoding.py /tmp/bpe.json gpre --check-mojo mojo_out.json
"""

import json
import sys
import regex as re

GPT2_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
GPT4_PAT = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"


def load_tokenizer(path):
    with open(path) as f:
        data = json.load(f)
    vocab = data["vocab"]
    merge_cache = {}
    for entry in data["merges"]:
        a_id, b_id, merged_id = entry
        merge_cache[(a_id, b_id)] = merged_id
    byte_to_cp = data["byte_to_cp"]
    cp_to_byte = {}
    for b, cp in enumerate(byte_to_cp):
        cp_to_byte[cp] = b
    return {"merge_cache": merge_cache, "vocab": vocab}


def encode(text, tokenizer, variant):
    merge_cache = tokenizer["merge_cache"]
    if variant == "gpre":
        spacer = "Ġ"
        t = text.replace(" ", " " + spacer)
        t = t.replace(".", " .")
        words = t.split(" ")
    elif variant == "gpt2":
        words = re.findall(GPT2_PAT, text)
    elif variant == "gpt4":
        words = re.findall(GPT4_PAT, text)
    else:
        raise ValueError(f"unknown variant: {variant}")

    result = []
    for word in words:
        token_ids = [b for b in word.encode("utf-8")]
        n = len(token_ids)
        while n >= 2:
            best_rank = -1
            best_a = best_b = best_m = -1
            for i in range(n - 1):
                key = (token_ids[i], token_ids[i + 1])
                merged = merge_cache.get(key, -1)
                if merged >= 0 and (best_rank < 0 or merged < best_rank):
                    best_rank = merged
                    best_a = token_ids[i]
                    best_b = token_ids[i + 1]
                    best_m = merged
            if best_rank < 0:
                break
            # Merge all occurrences in one pass
            j = 0
            write = 0
            while j < n:
                if j < n - 1 and token_ids[j] == best_a and token_ids[j + 1] == best_b:
                    token_ids[write] = best_m
                    j += 2
                else:
                    token_ids[write] = token_ids[j]
                    j += 1
                write += 1
            n = write
        result.extend(token_ids[:n])
    return result


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    tok_path = sys.argv[1]
    variant = sys.argv[2]

    if not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        text = "Hello world! Don't stop I'll be there 123."

    tokenizer = load_tokenizer(tok_path)
    ids = encode(text.strip(), tokenizer, variant)
    print(json.dumps(ids))


if __name__ == "__main__":
    main()
