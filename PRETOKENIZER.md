# Pretokenization: Current State vs tiktoken

## The Problem

Our `PreTokenizer` (`tokenizer.mojo`) uses a `Ġ` (U+0120) convention inherited
from the GPT-2 Python reference implementation (`encoder.py`): spaces at word
boundaries are replaced with `Ġ` during pre-tokenization, and decode must
reverse the substitution (`Ġ → 0x20`).

tiktoken (OpenAI's Rust tokeniser) **does not use `Ġ` at all**. Its regex-based
pretokenizer keeps whitespace bytes inline with the following token. Decode is
a trivial `Vec<u8>` concatenation.

## tiktoken GPT-2 / r50k_base Pattern

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
```

| # | Alternative | Behaviour |
|---|---|---|
| 1 | `'(?:[sdmt]\|ll\|ve\|re)` | Apostrophe contractions: `'s`, `'t`, `'m`, `'d`, `'ll`, `'ve`, `'re` |
| 2 | ` ?\p{L}++` | Optional ASCII space + one or more Unicode letters (possessive) |
| 3 | ` ?\p{N}++` | Optional ASCII space + one or more Unicode digits (possessive, no max) |
| 4 | ` ?[^\s\p{L}\p{N}]++` | Optional ASCII space + one or more non-whitespace, non-letter, non-digit characters (possessive) |
| 5 | `\s++$` | Trailing whitespace at end of string (possessive) |
| 6 | `\s+(?!\S)` | Whitespace run NOT followed by a non-space character (lookahead) |
| 7 | `\s` | Single whitespace character (fallback) |

All whitespace is kept as-is in the token. No `Ġ` substitution anywhere.

## tiktoken GPT-4 / cl100k_base Pattern (for reference)

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s
```

Key differences from GPT-2:
- Contractions are case-insensitive (`(?i:...)`)
- ` ?\p{L}++` → `[^\r\n\p{L}\p{N}]?+\p{L}++` (optional ANY non-category prefix, not just space)
- ` ?\p{N}++` → `\p{N}{1,3}+` (max 3 digits, no space prefix)
- `[\r\n]*+` suffix on punctuation rule
- Explicit newline rule: `\s*[\r\n]`

## `bpe.mojo` Pretokenizer Analysis

The existing pure-Mojo UTF-8 state machine in `../bpe.mojo/bpe/pretokenizer.mojo`
implements the **cl100k_base** pattern, **not** r50k_base. This causes lock-step
deviations from the GPT-2 tokenizer that our project targets:

### Matcher-by-matcher comparison

| Matcher | GPT-2 (what we need) | bpe.mojo (what it does) | Impact |
|---|---|---|---|
| `_match_contraction` | `'(?:[sdmt]\|ll\|ve\|re)` — case-sensitive | `'(?i:[sdmt]\|ll\|ve\|re)` — case-insensitive | Matches `'S`, `'T`, etc. where GPT-2 wouldn't |
| `_match_letter_run` | ` ?\p{L}++` — optional space + letters | `[^\r\n\p{L}\p{N}]?+\p{L}+` — optional ANY non-letter prefix | `"$hello"` is one chunk in bpe vs two in GPT-2; changes BPE boundary |
| `_match_digit_run` | ` ?\p{N}++` — optional space + digits, no max | `\p{N}{1,3}` — max 3 digits, no space prefix | `" 1234"` → `" 12"+"34"` in bpe vs `" 1234"` in GPT-2 |
| `_match_punct_run` | ` ?[^\s\p{L}\p{N}]++` — optional space + punct | Same + `[\r\n]*` suffix | Punctuation tokens eat trailing newlines |
| `_match_newline` | Not present | `\s*[\r\n]+` (cl100k rule) | Extra category not in GPT-2 |
| `\s++$` (alt 5) | Trailing whitespace only at EOF | Not implemented | Trailing spaces mid-string misclassified |
| `\s+(?!\S)` (alt 6) | Whitespace not followed by non-space | Not implemented (no lookahead) | Whitespace boundary differs from GPT-2 |
| `\s` (alt 7) | Single whitespace | Covered by `_match_whitespace` (ASCII-only) | OK for ASCII, misses Unicode whitespace |

### Minor issues

1. **Possessive quantifiers not possessive** — `?+`, `++`, `*+` in the regex
   prevent backtracking. The state machine's greedy-then-fail is equivalent in
   practice for these patterns (no backtracking needed when longest-match wins).

2. **Unicode whitespace** — `_match_whitespace` and `_match_newline` only check
   ASCII whitespace bytes (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20). Unicode
   whitespace codepoints (non-breaking space, ideographic space, etc.) will
   fall through to the single-codepoint fallback. This is fine for English text
   but a correctness gap for multilingual input.

3. **Digit run max of 3** — cl100k_base caps digit runs at 3 codepoints.
   GPT-2 has no such limit. This is the most visible difference for numeric
   text.

## Plan to Match GPT-2 / tiktoken Exactly

To bring our pretokenizer into lock step with tiktoken's GPT-2 pattern, we need to:

1. **Rewrite `_match_letter_run`** for ` ?\p{L}++`:
   - Optional ASCII space (byte 0x20) before letter run
   - Not `[^\r\n\p{L}\p{N}]?` prefix

2. **Rewrite `_match_digit_run`** for ` ?\p{N}++`:
   - Optional ASCII space before digit run
   - No max length (or large enough to never truncate practical input)
   - Possessive (greedy is equivalent here)

3. **Rewrite `_match_punct_run`** for ` ?[^\s\p{L}\p{N}]++`:
   - Remove trailing `[\r\n]*`

4. **Rewrite `_match_newline` → remove** (not in GPT-2 pattern)

5. **Rewrite `_match_whitespace`** with three sub-matchers for alternatives 5-7:
   - `\s++$` — whitespace at EOF only
   - `\s+(?!\S)` — whitespace not followed by non-space (approximate via byte lookahead)
   - `\s` — single whitespace fallback

6. **Remove `{\L}` lookahead commentary** — contractions should be case-sensitive
   for GPT-2 (GPT-4 adds `(?i:...)`).

After this, the pretokenizer split output will match tiktoken exactly, and we
can drop the `Ġ` convention entirely, eliminating the decode substitution
branch naturally (it's already gone via pre-substitution in `token_bytes`).
