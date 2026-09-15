---
title: "BPE Tokenization from Scratch in Mojo: A Beginner's Primer, with mbpe"
date: "2026-09-12"
categories: ["Machine Learning", "Mojo"]
tags: ["bpe", "tokenization", "mojo", "tiktoken", "nlp", "from-scratch"]
excerpt: "A ground-up primer on byte-pair encoding, built around mbpe — a from-scratch Mojo tokenizer that matches OpenAI's tiktoken byte-for-byte, and runs fast doing it."
---

# BPE Tokenization from Scratch in Mojo: A Beginner's Primer, with mbpe

A language model has never seen a single word in its life.

Feed it `hello world`, and before anything resembling "understanding" happens, those 11 characters get turned into a short list of integers — something like `[31373, 995]`. The model does math on those integers. At the very end, it converts integers back into text so a human can read the output. Every model you've used — GPT, Claude, Llama, all of them — is bracketed by this one step on either side, and almost nobody outside the field ever looks at it directly.

This post is a ground-up primer on that step, built around `mbpe`, a byte-pair-encoding tokenizer I wrote from scratch in Mojo. No prior tokenization knowledge assumed. By the end, you'll be able to trace a string of text all the way down to token IDs and back, by hand, and understand why each stage of that pipeline exists.

## Part 1: What Problem Is a Tokenizer Actually Solving?

Start with the constraint every tokenizer has to live inside: a model needs a **fixed-size vocabulary** — a finite table of "things it knows about" — and every piece of input text has to be rewritten as a sequence of entries from that table. Pick the table wrong and everything downstream suffers.

**Try option 1: one table entry per word.** This is how humans naturally think about language, so it feels like the obvious choice. It falls apart almost immediately. English alone has hundreds of thousands of words. Add proper names, typos, code identifiers like `resurrect_db_connection()`, emoji, and every other language on Earth, and there's no table size that covers it. Whatever doesn't fit gets mapped to a generic `<UNK>` ("unknown") token — and the model loses all information about what that word actually was. Train a word-level tokenizer on news articles, and it will choke the first time it sees a chunk of source code.

**Try option 2: one table entry per character, or per raw byte.** This fixes coverage completely. With roughly 256 possible byte values (plus some handling for multi-byte UTF-8 sequences), you can represent absolutely any input, in any language, with zero `<UNK>` tokens ever. But now sequences get long. The word `aroundtheworldineightydays` becomes 27 separate tokens instead of one or two. Longer sequences cost more compute (a transformer's attention mechanism scales with sequence length), and they bury useful structure: the model has to work harder to notice that `token` and `tokens` are related, because that relationship is now spread across many individual character-steps instead of showing up as one shared prefix.

**Byte-pair encoding (BPE) is the compromise**, and once you see the mechanism, it stops feeling like a compromise and starts feeling obvious. Common words stay as single whole tokens (`hello` is one token, not five). Rare or unfamiliar words get split into meaningful pieces (`tokenization` might become `token` + `ization`). Nothing is ever truly unknown, because the scheme can always fall back to raw bytes as a last resort.

The algorithm itself isn't new. It comes from a 1994 data-compression technique by Philip Gage, later repurposed specifically for word segmentation in machine translation by Sennrich, Haddow, and Birch in 2016. `mbpe` implements that same idea at the byte level, with no `<UNK>` possible by construction — any valid UTF-8 string that goes in, comes back out identical after an encode-then-decode round trip.

## Part 2: The Whole Algorithm, in One Worked Example

Forget code for a moment. Here is the entire BPE algorithm, traced by hand on a toy corpus, so the mechanism is completely concrete before a single line of Mojo shows up.

Take the string `aaabdaaabac`. This is a real example used in this project's own test suite, chosen because it's small enough to trace on paper.

**Step 0 — start from raw symbols.** Every character is its own token to begin with: `a`, `a`, `a`, `b`, `d`, `a`, `a`, `a`, `b`, `a`, `c`.

**Step 1 — count every adjacent pair.** Walk through the sequence and tally which two-token combinations appear next to each other:

| pair | count |
|---|---|
| `aa` | 4 |
| `ab` | 2 |
| `bd` | 1 |
| `da` | 1 |
| `ba` | 1 |
| `ac` | 1 |

**Step 2 — merge the winner.** `aa` appears most often, so it becomes a brand-new token — call it `Z` — and every occurrence of `aa` in the sequence gets replaced by `Z`. The sequence is now: `Z`, `a`, `b`, `d`, `Z`, `a`, `b`, `a`, `c`.

**Step 3 — repeat.** Count pairs again in this rewritten sequence. Now `Za` shows up twice — merge it into a new token `Y`. Rewrite again: `Y`, `b`, `d`, `Y`, `b`, `a`, `c`.

**Step 4 — repeat once more.** `Yb` now shows up twice. Merge it into a new token `W`. Rewrite again: `W`, `d`, `W`, `a`, `c`.

Three merges learned, in this exact order:

| rank | rule | what it means in bytes |
|---|---|---|
| 256 | `aa` → `Z` | `Z` = `"aa"` |
| 257 | `Za` → `Y` | `Y` = `"aaa"` |
| 258 | `Yb` → `W` | `W` = `"aaab"` |

That loop — *count pairs, merge the most frequent one, repeat* — is training. It needs the whole corpus in front of it, because it has to count. Every other section of this post is really just answering one question about that loop: how do you make it correct, and how do you make it fast?

### Encoding new text doesn't repeat any of that

Here's the part that trips people up: encoding a brand-new piece of text later does **not** re-run this counting process. There's no corpus to count over anymore — often there's just one short string, sometimes just one word. Counting pair frequencies in an 8-character input would be meaningless; it isn't a statistical sample of anything.

Instead, encoding reuses the merge *rules* discovered above, in the exact order they were discovered — rank 256 before 257 before 258 — and applies them to the new text's raw bytes, repeatedly, until nothing more matches.

Walk it by hand. Encode the string `aaab` — note this exact substring also happens to appear in the training corpus, so we already know what the "correct" answer should look like.

**Start from raw bytes:** `a`, `a`, `a`, `b`.

**Pass 1 — look for the lowest-rank rule that matches anywhere in this sequence.** Rank 256 (`aa` → `Z`) matches at positions 1–2. No lower-ranked rule is available, so apply it, merging every non-overlapping occurrence left to right: `Z`, `a`, `b`.

**Pass 2 — check again from the top.** Rank 256 no longer matches (no `aa` pair left). Rank 257 (`Za` → `Y`) matches at position 1–2: `Z` followed by `a`. Apply it: `Y`, `b`.

**Pass 3 — check again.** Rank 258 (`Yb` → `W`) matches at position 1–2: `Y` followed by `b`. Apply it: `W`.

**Pass 4 — check again.** One token left, no pairs at all. Stop.

`aaab` encodes to the single token `W` — exactly the token the training pass built for that exact substring. Four raw bytes, one ID, zero counting, three lookups.

### Why the order has to be rank order, not just "any matching rule"

It's tempting to think any rule that matches can be applied whenever you spot it. It can't — and the reason isn't that decoding would break. It's that encoding would stop being deterministic.

Suppose, instead of respecting rank order, you'd applied rank 257 wherever convenient and rank 256 only afterward. On `aaab`, that's actually impossible to do out of order here — `Y` doesn't exist until `Z` exists, which doesn't exist until rule 256 has fired. The dependency enforces itself.

But consider a case without that built-in dependency: imagine a fourth learned rule, `ab` → `V`, sitting at some rank. On the input `aab`, two different valid-looking merges are available at the very first step: `aa` (rule 256) at positions 1–2, and `ab` (the hypothetical rule) at positions 2–3 — they overlap, so only one can be taken first. Take `aa` first and you get `Z`, `b`. Take `ab` first and you get `a`, `V`. Both decode back to `"aab"` correctly — decoding never breaks, because it's just concatenating bytes. But they are two *different* token-ID sequences for the same string, and only one of them is what the model that consumed this tokenizer's output was actually trained on.

That's the entire reason rank order is non-negotiable at encode time: it's what makes the mapping from text to IDs a function rather than a coin flip. Always resolve ties and choices by picking the lowest available rank — the rule discovered earliest, because "discovered earliest" is just another way of saying "was the single most frequent pattern in the training data at that point." Anything else and the same input string could legally tokenize two different ways depending on implementation mood, and `tiktoken` parity — matching another tokenizer's IDs exactly — would be impossible by construction, not just by bug.

This is also exactly what `_merge_scan` in `bpe/tokenizer.mojo` is doing under the hood during real encoding (Stage 5, below): repeatedly find the lowest-ranked mergeable pair present anywhere in the word, merge every occurrence of it, and repeat — the same three passes just walked above, just running in compiled Mojo instead of on paper.

Here's that same loop, but on real English text, run against this actual codebase:

```python
import mbpe
tok = mbpe.get_encoding("gpt2")
tok.encode("hello world")   # [31373, 995]
tok.decode([31373, 995])    # 'hello world'
```

Eleven characters collapse into two integers, because `hello` and ` world` (note: the leading space is part of the token — more on this below) each survived GPT-2's training as a single whole token. `tok.n_vocab` reports `50257` — that's 256 starting bytes, plus 50,000 learned merges, plus one special end-of-text marker. Every one of these exact numbers has to match `tiktoken`, OpenAI's own tokenizer library, precisely — a dedicated parity test suite in this repo checks that on every code change.

## Part 3: A Map of the Codebase

```
bpe/shared.mojo          # ByteArray, TokenSpan, ByteSpanArena (storage)
bpe/tokenizer_trait.mojo # minimal Tokenizer trait (encode/decode)
bpe/pretokenizer.mojo    # PreTokenizer trait + GPT2 / GPT4 variants
bpe/tokenizer.mojo       # BPETokenizer[PT]: train, encode, decode, save/load
python-binding/mbpe.mojo # Mojo -> Python shared library (_mbpe.so)
python-binding/mbpe/__init__.py  # tiktoken-compatible Python wrapper
main.mojo                # Mojo test entrypoint (33 tests)
tests/                   # test_tokenizer.mojo, exhaustive_tokenizer.mojo, tests/python/
data/                    # gpt2.tiktoken, cl100k.tiktoken, o200k.tiktoken (+ variants)
```

Read in this order — `shared.mojo` → `pretokenizer.mojo` → `tokenizer.mojo` → `python-binding/` — and you're following data through the system in the same order it actually flows at runtime: input, then processing, then output. The sections below follow the same order.

## Stage 1: Storage — How a Token Actually Lives in Memory

A token, underneath everything, is just a run of bytes. Token 72 might be `b"h"`. Token 31373 might be `b"hello"`. The tokenizer needs to store tens of thousands of these variable-length byte strings and look any one up by its ID, fast — this matters most during decoding, where speed is the whole game.

The obvious approach — a plain list of strings, indexed by ID — works, but it means every single token pays for its own separate memory allocation. Multiply that by 50,000+ tokens, and both memory use and CPU cache behavior suffer.

`bpe/shared.mojo` solves this in 79 lines with a different structure:

```mojo
comptime IntArray = List[Int]
comptime ByteArray = List[Byte]

@fieldwise_init
struct TokenSpan(TrivialRegisterPassable & Equatable & Writable):
    var offset: Int
    var length: Int
```

A `TokenSpan` doesn't hold any bytes itself — it holds an `offset` and a `length` pointing into one single, large, shared byte buffer. This pattern is called an **arena**. Instead of many small allocations, you make one large allocation up front and hand out lightweight "claim tickets" into it.

A coat-check is the right mental picture: all the coats (bytes) live on one rack. Your ticket (the span) just says "yours starts at hook 40, and it's 5 coats long." Handing someone a ticket is nearly free; handing someone their own private coat closet is not.

`ByteSpanArena` is that rack:

```mojo
struct ByteSpanArena(ImplicitlyCopyable & Sized & Writable):
    var bytes: ByteArray
    var spans: List[TokenSpan]
```

Registering the token `b"hello"` means appending its 5 bytes to `bytes`, and pushing `TokenSpan(offset, 5)` onto `spans`. Decoding ID 31373 later means reading that one span and copying exactly that slice back out — one lookup, one slice, zero per-token allocation, zero hashing.

`TokenByteTable`, defined in `bpe/tokenizer.mojo:100`, wraps this arena into something with clean ID semantics — think of it simply as `id -> bytes`:

```mojo
struct TokenByteTable(ImplicitlyCopyable & Sized & Writable):
    var arena: ByteSpanArena
```

Once the coat-check picture clicks, decoding (Stage 6, below) will feel obvious before you even get there — it's the identical "hand over the ticket, get back the coat" operation.

## Stage 2: Pre-tokenization — Split Before You Merge

### Why this step can't be skipped

BPE's merge step only ever merges *adjacent* tokens. But adjacent according to what boundary? Feed an entire sentence in as one undifferentiated stream of bytes, and the trailing `he` in `the` looks identical to the algorithm as the leading `he` in `hello` in the phrase `the hello`. Left unchecked, merges would bleed across word boundaries in ways that make no linguistic sense and generalize terribly to new text.

Pre-tokenization solves this by chopping text into words and word-like chunks *before* any merging happens — using splitting rules that specifically mirror what GPT-2 and GPT-4 do. This is most of what "tiktoken-compatible" means in practice(and we have to jump quite a few hoops to make that happen!): not just "the same algorithm," but "the same chunk boundaries," down to the character.

See it directly, real output from this repo:

```python
tok2 = mbpe.GPT2Tokenizer()
tok2.train(['hello world', 'hello there'], 300)
tok2._tok.pretokenize("Hello world! Don't stop")
# [b'Hello', b' world', b'!', b' Don', b"'t", b' stop']
```

Two details worth sitting with here if this is new to you. First, `' world'` keeps its *leading* space glued to the token — GPT-2-style tokenizers attach whitespace to the word that follows it, so `hello` and ` world` are different token identities even though they share four of the same letters. Second, `Don't` splits as `b' Don'` + `b"'t"`, not the more "natural-looking" `b' Do'` + `b"n't"`. That exact split isn't an accident — it mirrors GPT-2's own splitting rule precisely, because the resulting IDs must match `tiktoken` token-for-token. A different split point here would silently produce different, wrong IDs downstream, with no error to warn you.

### The code

`bpe/pretokenizer.mojo:357` lays out the contract every pre-tokenizer must satisfy:

```mojo
trait PreTokenizer(Movable & Defaultable & Deinitable & Writable):
    comptime byte_map: ByteMapping
    # ...
    def split_view(self, text: StringSlice) raises -> List[StringSlice]:
    def split(self, text: StringSlice) raises -> List[String]:
    def count_words(self, text: StringSlice, mut counts: WordCounts) raises:
    @staticmethod
    def name() -> String: ...
    @staticmethod
    def special_tokens() -> Dict[String, Int]: ...
```

`PT` here is a *compile-time* parameter, not a runtime flag. `BPETokenizer[GPT2Pretokenizer]` and `BPETokenizer[GPT4Pretokenizer[...]]` are genuinely different types in Mojo's eyes, each with its splitting rule baked directly into the compiled binary. There's no `if model == "gpt2": ...` branch sitting in a hot loop, checked millions of times per encode call — the compiler already resolved that choice once, at build time.

Two implementations ship today:

- **`GPT2Pretokenizer`** (`bpe/pretokenizer.mojo:457`) recognizes letter runs, digit runs, punctuation runs, the specific contractions `'t`, `'ll`, `'ve`, `'re`, and whitespace. Its core dispatcher, `_best_match` (line 600), tries each rule in order: contraction, then letters, then digits, then punctuation, then whitespace.
- **`GPT4Pretokenizer[mapping]`** (`bpe/pretokenizer.mojo:663`) follows the same overall shape, but differs in details — how digit runs like `123` are handled, for instance. It also takes a `mapping` parameter, the subject of Stage 3.

Neither implementation reaches for a regex engine. Both are hand-written byte scanners, walking a raw byte buffer directly with helpers like `_match_letter_run`, `_match_digit_run`, and `_match_punct_run`, backed by lookup tables in `bpe/unicode_tables.mojo`. That's a deliberate trade: more code up front, in exchange for never paying regex-engine overhead on every split of every word, for the life of the tokenizer. It's also why `main.mojo` includes split-alignment tests like `test_gpt2_splits` and `test_gpt4_splits` — hand-rolled scanners need something pinning down exact boundaries, since there's no regex source-of-truth to defer to.

One Mojo-specific wrinkle worth knowing: `split_view` returns zero-copy *views* into the original text, allocating nothing new. `split` copies those views into owned strings, which does allocate. Training deliberately uses `count_words`, which counts directly off views without ever allocating a string — a distinction that sounds pedantic until you're training on megabytes of text and that "unnecessary" allocation is now happening millions of times.

## Stage 3: The Odd One Out — o200k Shuffles Its Bytes

For GPT-2 and `cl100k` (GPT-4's older tokenizer), the mapping from a raw byte value to its starting token ID is trivial: byte value 97 (the character `a`) simply *is* token ID 97. Nothing to compute.

GPT-4o's `o200k` encoding breaks that pattern on purpose. It permutes the mapping — rank 0 turns out to be `!` (byte `0x21`), the space character (`0x20`) lands at rank 220, and `a` (`0x61`) lands at rank 64. Assume "byte value = starting ID" while working with `o200k`, and every ID you produce will be wrong — not obviously wrong, just silently, plausibly-shaped wrong. That's the worst kind of bug: one that never throws an error.

`bpe/pretokenizer.mojo:153` makes this explicit as its own type, rather than leaving it an implicit assumption buried in some function:

```mojo
struct ByteMapping(ImplicitlyCopyable & Equatable):
    var _value: Int
    comptime SEQUENTIAL = ByteMapping(0)
    comptime SHUFFLED = ByteMapping(1)
```

`GPT4Pretokenizer[SEQUENTIAL].byte_to_id(b)` just hands back `b` unchanged. `GPT4Pretokenizer[SHUFFLED].byte_to_id(b)` looks the answer up in a precomputed table instead. Because that choice is a compile-time parameter, the branch selecting between the two behaviors disappears entirely from the compiled binary — a sequential-mapping build pays nothing, not even a branch check, for shuffle support it will never use.

A cheat sheet, since these three are genuinely easy to mix up by name alone:

| Python class | Mojo type | File name | Byte map |
|---|---|---|---|
| `GPT2Tokenizer` | `BPETokenizer[GPT2Pretokenizer]` | `gpt2.tiktoken` | sequential |
| `GPT4Tokenizer` | `BPETokenizer[GPT4Pretokenizer[SEQUENTIAL]]` | `cl100k.tiktoken` | sequential |
| `GPT4oTokenizer` | `BPETokenizer[GPT4Pretokenizer[SHUFFLED]]` | `o200k.tiktoken` | shuffled |

Same input string, three different — and correct, by design — outputs:

```python
mbpe.get_encoding("gpt2").encode("hello world")    # [31373, 995]
mbpe.get_encoding("cl100k").encode("hello world")  # [15339, 1917]
mbpe.get_encoding("o200k").encode("hello world")   # [24912, 2375]
```

If you're ever debugging `o200k` output that *looks* plausible but is subtly off, check `ByteMapping` before anything else. It's the easiest mistake to make anywhere in this codebase.

## Stage 4: Training — Counting Pairs and Merging Winners, at Scale

Part 2 walked the algorithm by hand on 11 characters. This stage is the same loop, run on real text, with the engineering needed to make it finish in a reasonable amount of time.

A small, real training run against this repo:

```python
tok2 = mbpe.GPT2Tokenizer()
tok2.train(['hello world', 'hello there'], 300)
tok2.encode('hello world')  # [259, 267] — small IDs, because the vocab is tiny
```

With the full 50,257-entry GPT-2 vocabulary, `hello` collapses all the way down to one single token (`31373`). With this toy 300-entry vocabulary trained on just two sentences, it's still stitched together from a couple of pieces — same algorithm, just stopped far earlier, with far fewer opportunities to have decided `hello` deserved its own slot.

`bpe/tokenizer.mojo:248` is where training begins:

```mojo
def train(mut self, corpus: Span[String], vocab_size: Int) raises:
    if vocab_size < 256:
        raise Error("vocab_size must be at least 256 ...")
    var word_counts = WordCounts()
    for text in corpus:
        self.pt.count_words(text, word_counts)
```

Step one counts words — really, pre-tokenized chunks — along with how often each occurs. From there it builds three parallel structures: an `arena` holding every word's bytes as rank IDs, offset/length/frequency arrays describing where each word lives, and a pair-frequency map (`stats`) alongside an index of which words contain which pairs (`where_dict`).

The main loop (`bpe/tokenizer.mojo:318`) is the textbook algorithm made concrete:

```mojo
while len(self.token_table) < vocab_size:
    # find most frequent pair in stats
    # ...
    if max_freq <= 0:
        break
    # merge every occurrence in affected words only
    # append MergeRule(a_id, b_id, merged_id)
    # update stats incrementally
```

Two choices here are doing the real work of turning "correct" into "fast":

**Incremental statistics.** A textbook-literal implementation recomputes pair counts over the *entire* corpus after every single merge — roughly O(vocabulary size × corpus size) total work, which turns painfully slow after a few thousand merges. This implementation instead only revisits the words that actually contained the just-merged pair, updating just the neighboring counts those words touch. That's the difference between training finishing in seconds and not finishing in a reasonable session at all.

**Two-tier lookup.** A `MergeRule` is a simple triple — `first`, `second`, `merged` — and `MergeLookup` stores these in a flat array for small IDs, falling back to a dictionary only for the rest:

```mojo
struct MergeLookup(ImplicitlyCopyable & Writable):
    var _fast: IntArray   # CACHE_ENTRIES flat array
    var _slow: Dict[Int, Int]
```

`get(id1, id2)` checks the flat array directly when both IDs are under 1000, only falling through to the (slower) hash-based dictionary otherwise. Encoding performs millions of these lookups over the life of a program, so keeping the common case — small, frequently-seen IDs — off the hashing path is a real, measurable win.

## Stage 5: Encode — From Text to IDs

The full pipeline, in order:

```
text -> split into words (PreTokenizer, Stage 2)
     -> bytes -> ranks (byte_to_id, Stage 3)
     -> apply learned merges lowest-rank-first until stuck
     -> append IDs to result
```

`encode_ordinary` (`bpe/tokenizer.mojo:530`) is the hot path used when no special tokens are involved. For each pre-tokenized word, it picks one of two merge strategies based on word length:

```mojo
if n < 2:
    self._copy_word_ids(ptr, n, dst)
elif n < SCAN_LIMIT:   # 32
    write_pos += self._merge_scan(ptr, n, dst)
else:
    write_pos += self._merge_heap(ptr, n, dst, scratch, heap)
```

`_merge_scan` is deliberately simple: repeatedly find the lowest-ranked mergeable pair anywhere in the word, and merge every occurrence of it. That's technically O(n²) in word length — which sounds alarming until you remember that almost all real words are well under 10 characters, so the simple version wins outright at that scale.

`_merge_heap` exists for the rare long word — one built from a `BinaryHeap` of candidate merges keyed by rank, backed by a linked-list representation so each merge is an O(log n) heap update instead of a full word rescan. Short words use the scanner; long words use the heap. Same final answer either way — it's purely a speed decision, invisible from outside.

`encode` itself wraps `encode_ordinary` with special-token scanning layered on top, emitting a special token's ID directly wherever one is found in the input and encoding everything else as ordinary text. When no specials are registered, it skips straight through to `encode_ordinary` at effectively zero cost — locked down by a dedicated test (`test_special_tokens_no_specials_registered`) so that supporting specials never taxes the far more common case of not using any.

## Stage 6: Decode — From IDs Back to Text

Decode is the easy direction, and it's worth appreciating exactly how much easier: look up each ID's bytes, concatenate them, done. No merges to replay, no pre-tokenizer involved, no heap.

```mojo
def decode(self, ids: Span[Int]) raises -> String:
    # 1. validate range, sum total length
    # 2. String(unsafe_uninit_length=total)
    # 3. unsafe_memcpy each token's bytes into place
```

That asymmetry — decode is a pure memory-copy chain, encode has to run a merge search — is exactly why decode throughput measures roughly 10x encode throughput on this implementation (184.5M tokens/sec versus 17.7M tokens/sec for GPT-2, native Mojo, no Python in the loop). That's not a tuning accident; it's structural. Decode simply has less work to do, by the nature of the problem.

A few siblings live alongside `decode`: `decode_bytes` (same thing, returns raw bytes instead of a string), `decode_single_token_bytes(id)` (bytes for exactly one token — used to sanity-check merges), and `decode_with_offsets` (decoded text plus per-token start/end byte positions, useful for anything that needs to map model output back onto spans of the original text, like highlighting or citations).

## Stage 7: Files — the `.tiktoken` Format and Merge Recovery

A `.tiktoken` file on disk is nothing exotic — plain text lines of `base64(bytes) rank`:

```
IQ== 0
Ig== 1
Iw== 2
...
```

The first lines decode to single bytes. Later lines are multi-byte merge products, in the exact order the training loop discovered them. Real line counts from this repo's data files:

```
50256 data/gpt2.tiktoken
100256 data/cl100k.tiktoken
199998 data/o200k.tiktoken
```

**Saving** walks the token table, skips any registered special tokens, base64-encodes each token's bytes, and writes one `encoded + " " + id` line per token. Two things follow from that: saves are fully deterministic (the same tokenizer state always produces a byte-identical file — useful if you ever want to diff two training runs), and special tokens are **not** written to the file at all. That second point is the single most common thing that trips people up, and it comes back in the pitfalls list below.

**Loading** reads the file back and rebuilds the token table — but the file only records *what* each token is, not the merge history that produced it (which two earlier tokens combined to form it). Encoding needs that history, in priority order, to work at all. So `_recover_merges` reconstructs it: for every token at or above ID 256, it reruns the merge search restricted to lower-ranked tokens, discovers which two pieces merged last to produce it, and records that as a `MergeRule`. A dedicated test checks the resulting invariant holds for every recovered merge: `bytes[merged] == bytes[left] + bytes[right]`.

After recovery, any special tokens defined by the pre-tokenizer type itself get automatically re-registered — which is why loading `gpt2.tiktoken` always ends with `<|endoftext|>` mapped to `50256`, even though the file on disk never mentions it explicitly. Any *custom* special tokens you added yourself have to be re-registered by hand after every reload.

## Stage 8: Python Bindings — Mojo Speed, tiktoken's API

Most people who use this library will never write a line of Mojo. They'll write this instead:

```python
import mbpe
tok = mbpe.get_encoding("gpt2")
tok.encode("hello world", allowed_special={"<|endoftext|>"})
```

That single call passes through two layers. The first, `python-binding/mbpe.mojo`, compiles down to a shared library (`_mbpe.so`) and defines three fully-specialized concrete types — one each for GPT-2, GPT-4 (`cl100k`), and GPT-4o (`o200k`) — each exposed to Python with `train`, `encode`, `decode`, `save/load_tiktoken`, and more.

The second layer, `python-binding/mbpe/__init__.py`, wraps that compiled library to match `tiktoken`'s actual Python API. The Mojo layer only understands two encode modes: special-token-aware, or "treat everything as plain text." All of `tiktoken`'s richer `allowed_special` / `disallowed_special` handling lives entirely in this Python wrapper, which — for the partial-allowed-set case — manually splits the input around the allowed specials and calls the plain-text encoder on each gap in between.

The distinction between the two encode modes shows up clearly in real output:

```python
tok.encode("<|endoftext|>hello world")      # [50256, 31373, 995]
tok.encode_ordinary("<|endoftext|>hello")   # [27, 91, 437, 1659, 5239, 91, 29, 31373]
```

`encode` recognizes the special marker and emits it as one clean ID. `encode_ordinary` has no concept of "special" at all, and shreds the exact same text into eight separate tokens, byte by byte and merge by merge. This is the single most common point of confusion for anyone new to `tiktoken`-style tokenizers generally, not just this codebase.

## Put It All Together: A Script That Touches Every Stage

Every output below is real, copied directly from this repo — nothing hypothetical.

```python
import sys
sys.path.insert(0, 'python-binding')
import mbpe

# Load a pretrained tokenizer (Stage 7 + Stage 8)
tok = mbpe.get_encoding("gpt2")
print(tok.encode("hello world"))      # [31373, 995]
print(tok.decode([31373, 995]))       # 'hello world'
print(tok.n_vocab)                    # 50257

# Train a tiny tokenizer from scratch (Stage 4)
tiny = mbpe.GPT2Tokenizer()
tiny.train(["hello world", "hello there"], 300)
print(tiny.encode("hello world"))     # [259, 267]

# Inspect pre-tokenization directly (Stage 2)
print(tiny._tok.pretokenize("Hello world! Don't stop"))
# [b'Hello', b' world', b'!', b' Don', b"'t", b' stop']

# Compare across encodings (Stage 3)
print(mbpe.get_encoding("cl100k").encode("hello world"))  # [15339, 1917]
print(mbpe.get_encoding("o200k").encode("hello world"))   # [24912, 2375]
```

The same input string produces genuinely different ID sequences under each encoding — different pre-tokenizer, different learned merges, and (for `o200k`) a different byte map — yet every single one decodes back to the exact original text. Training a fresh tokenizer takes two lines. The tiny vocabulary above only produces small IDs because it had just 44 merges to learn (300 − 256) from two short sentences.

The Mojo-native equivalent, for anyone skipping Python entirely:

```mojo
from bpe.tokenizer import Tokenizers

def main() raises:
    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    var ids = gpt2.encode("hello world")
    print(gpt2.decode(ids))  # hello world
```

`Tokenizers.gpt2` is a compile-time alias for `GPT2Pretokenizer`. Swap in `Tokenizers.o200k` and every other line stays identical — no branches to add, nothing else to change. That's the entire payoff of making the pre-tokenizer a compile-time parameter instead of a runtime string: switching encodings is a one-token edit, not a refactor.

## Is It Actually Fast?

It's fast — not by assertion, but by benchmark, run against Python `tiktoken` and against `tiktoken-rs` (the Rust reimplementation OpenAI itself uses in production) across the `gpt2`, `cl100k`, and `o200k` encodings, at multiple vocabulary sizes, on both encode and decode.

On native Mojo, decode alone runs at roughly 184.5 million tokens per second for GPT-2, versus roughly 17.7 million tokens per second for encode on the same encoding — the asymmetry Stage 6 explains structurally. Across the benchmark suite, native Mojo leads or ties `tiktoken-rs` on every encode/decode row, and the Python bindings outperform Python `tiktoken` itself on the `gpt2` and `cl100k` encodings. I'm not going to claim this is the single fastest BPE implementation that exists anywhere — I haven't benchmarked every implementation on Earth, and it would be a strange thing to assert without doing that work. What I can say plainly: on the comparisons I've actually run, against the two most relevant reference implementations, this one is at least as fast, and often faster.

## Common Pitfalls (Read Before You Debug)

1. **Stale compiled library.** Python imports the *compiled* `_mbpe.so`, not your edited `.mojo` source directly. Change Mojo code, and if Python's behavior stubbornly doesn't change, you almost certainly forgot to rebuild.
2. **Wrong byte mapping for `o200k`.** Using the sequential mapping against an `o200k.tiktoken` file loads without any error — it just silently produces wrong IDs. If `o200k` output looks "almost right but shifted," this is the first thing to check.
3. **Specials vanish across save/load.** Saving intentionally skips them (Stage 7). After loading, only the specials built into the pre-tokenizer type come back automatically — any custom ones need to be re-registered by hand.
4. **`encode` vs. `encode_ordinary` confusion.** If a special token comes back as several separate token IDs instead of one clean ID, you've either called the wrong method, or called `encode` with an empty allowed-special set. Both are correct, intentional behavior — not a bug.
5. **Data directory mismatches.** The Mojo path and the Python path resolve their data directory differently. Setting an environment variable to fix one does nothing for the other.

## Takeaways

Start from one simple, slightly uncomfortable fact — models see integers, not text — and you can build all the way up to three interchangeable, from-scratch tokenizers sharing one engine underneath. The pre-tokenizer decides where chunk boundaries fall. Training counts adjacent pairs and merges the winners, cached for fast lookup. Encoding replays those learned merges per word, choosing a scan-or-heap strategy based on word length. Decoding, by contrast, is nothing more than a memory copy over an arena — genuinely the easy direction. Files on disk store base64-encoded bytes plus ranks, and the merge history encoding needs gets reconstructed on load rather than stored redundantly.

If you want this to actually stick rather than just be something you read: train a 300-vocabulary tokenizer on two sentences of your own, save it, reload it, and confirm `encode` gives you identical IDs before and after. That one loop touches every stage in this post. Once it feels routine, you understand BPE better than most people who use a tokenizer every day without ever looking inside one.

---

*Further reading: Sennrich et al. (2016), "Neural Machine Translation of Rare Words with Subword Units" (ACL 2016), for the original BPE-for-NLP adaptation; the Hugging Face LLM Course, Chapter 6.5, for a shorter walkthrough from a different angle; `README.md` and `CONTRIBUTING.md` in the [mbpe repository](https://github.com/ratulb/mbpe) for exact build and test commands.*
