---
title: "BPE Tokenization from Scratch in Mojo: A Code-First Tour of mbpe"
date: "2026-09-12"
categories: ["Machine Learning", "Mojo"]
tags: ["bpe", "tokenization", "mojo", "tiktoken", "nlp", "from-scratch"]
excerpt: "Learn byte-pair encoding from zero using mbpe, a Mojo tokenizer that matches OpenAI tiktoken — we trace training, pre-tokenization, encode/decode, and Python bindings with real code and real output."
---

# BPE Tokenization from Scratch in Mojo: A Code-First Tour of mbpe

If you have never thought about tokenization, start here: a language model does not see text. It sees integers.

When you type `hello world`, something must turn those 11 characters into a short list like `[31373, 995]`. That something is a tokenizer. The model looks up each integer in a table, does math on vectors, then converts integers back to text at the end. Every GPT-style model you have used — ChatGPT, Claude, Llama, all of them — starts and ends with exactly this step, and almost nobody outside the field ever sees it happen.

This post is a tour of `mbpe`, a complete byte-pair-encoding tokenizer written from scratch in Mojo that is bit-for-bit compatible with OpenAI's `tiktoken`. I am not going to claim it is the fastest tokenizer in existence or the most feature-complete — it isn't, and there is no need to pretend otherwise. What I can claim, and back with real numbers below, is that it is correct: it matches `tiktoken`'s IDs exactly across three encodings, it round-trips every valid UTF-8 string without loss, and every design decision in it is one I can point to and defend. That is a smaller claim than "fastest," and a truer one.

By the end of this post you will understand the whole pipeline — not just what BPE is in the abstract, but which file does what, why each stage exists, and where the sharp edges are if you go implement one yourself.

> **Note on structure.** This codebase was not built as staged tutorial files where each file is a "lesson." It is a pipeline: text → pre-tokenizer → BPE merges → IDs → bytes → text. So instead of touring stage-per-file in isolation, we follow data in the order it actually flows through the system at runtime. That is also the order I'd recommend reading the source in.

## The Hook: Why Not Just Use Words or Characters?

Before BPE, it's worth sitting with the problem it solves, because the solution looks arbitrary until you've felt the problem.

Say you want to give a model a fixed-size vocabulary — a finite table of "things it can recognize" — and every piece of text needs to become a sequence of entries from that table.

**Option 1: one entry per word.** This feels natural; it's how we think about language. But it breaks immediately at scale. English alone has hundreds of thousands of words, before you count names, typos, code identifiers, emoji, and every other language on earth. You cannot fit all of that into a table of, say, 50,000 entries. Whatever doesn't fit becomes `<UNK>` — "unknown" — and the model loses that information entirely. A word-level tokenizer trained on news articles will choke on `resurrect_db_connection()` from your codebase.

**Option 2: one entry per character (or byte).** This fixes coverage completely — with roughly 256 byte values plus some UTF-8 handling, you can represent literally any input, in any language, with no `<UNK>` ever. But it creates the opposite problem: sequences explode in length. `aroundtheworldineightydays` becomes 34 separate tokens instead of one or two. Long sequences are slower to process (attention cost grows with sequence length), and worse, they make it harder for the model to notice that `token` and `tokens` are obviously related words — that relationship is now buried across many individual character-steps instead of visible as one shared prefix-token.

**Byte-pair encoding is the compromise**, and once you see it, it feels almost obvious. Keep common words as single whole tokens (`hello` is one token, not five). Split rare or unfamiliar words into meaningful pieces (`tokenization` might become `token` + `ization`). Never produce `<UNK>`, because you can always fall back all the way down to raw bytes. The idea traces back to a 1994 data-compression algorithm by Philip Gage, later adapted specifically for word segmentation by Sennrich, Haddow, and Birch in 2016. The Hugging Face LLM course puts it cleanly: start from characters or bytes, then recursively merge the most frequent adjacent pairs into new, bigger units.

`mbpe` implements exactly that algorithm, at the byte level, with no `<UNK>` possible by construction. Any valid UTF-8 input round-trips through it exactly — encode then decode always gives you back the original bytes.

## The 30-Second Mental Model

Here is all of BPE in one paragraph, because everything else in this post is really just speed, file-format, and compatibility detail layered on top of this one idea.

Start with 256 tokens, one per possible byte value (0–255). Scan your training corpus and count which *pairs* of adjacent tokens show up most often. Take the single most frequent pair and merge it into a brand-new token — this gets the next free ID, 256. Rewrite the corpus with that merge applied. Repeat: count pairs again, merge the new winner, assign the next ID (257, 258, ...), and keep going until you hit your target vocabulary size — maybe 300 for a toy example, or 50,257 for GPT-2's real vocabulary. To encode brand-new text later, split it into small chunks, start from raw bytes, and apply your *learned* merges in the order you learned them (lowest new ID first, since that's the order they were discovered and therefore the order they should be reapplied).

That's the whole algorithm. Keep that loop in your head — count pairs, merge the winner, repeat — as we walk through the actual code, because every file below exists to make that loop either correct or fast.

Here is that loop in action, real output from this repo, not a hypothetical:

```python
import mbpe
tok = mbpe.get_encoding("gpt2")
tok.encode("hello world")   # [31373, 995]
tok.decode([31373, 995])    # 'hello world'
```

Two integers represent 11 characters, because `hello` and ` world` (note the leading space — more on that below) both survived training as whole tokens. `tok.n_vocab` is `50257`. These exact numbers have to match `tiktoken`'s output precisely, and the parity test suite enforces that on every change.

## Map of the Codebase: Only 6 Files That Matter

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

If you read in order, read `shared.mojo` → `pretokenizer.mojo` → `tokenizer.mojo` → `python-binding/`. That is input → processing → output, and it mirrors the section order below.

## Stage 1: Storage — How Tokens Live in Memory

### The idea

A token is, underneath everything, just a run of bytes. Token 72 might be `b"h"`. Token 31373 might be `b"hello"`. We need a way to store tens of thousands of these variable-length byte strings and look any one of them up by ID quickly — this matters most during decode, which needs to run fast.

The naive approach — a `List[String]` indexed by ID — works but means every token pays the overhead of being its own separately-allocated, separately-managed string object. For 50,000+ tokens that adds up in both memory and cache behavior.

### The code

`bpe/shared.mojo` defines the whole storage layer in 79 lines:

```mojo
comptime IntArray = List[Int]
comptime ByteArray = List[Byte]

@fieldwise_init
struct TokenSpan(TrivialRegisterPassable & Equatable & Writable):
    var offset: Int
    var length: Int
```

A `TokenSpan` does not hold bytes itself. It holds an `offset` and a `length` pointing into one single, big, shared `bytes` array. This pattern is called an **arena**: instead of many small allocations, you make one large allocation and hand out lightweight "claim tickets" (offset + length) into it. Think of it like a coat check — the coats (bytes) all live in one rack, and each ticket (span) just says "yours starts at hook 40 and is 5 coats long."

`ByteSpanArena`, in the same file, is the arena itself:

```mojo
struct ByteSpanArena(ImplicitlyCopyable & Sized & Writable):
    var bytes: ByteArray
    var spans: List[TokenSpan]
```

To register the token `b"hello"`, we append its 5 bytes to `bytes` and push `TokenSpan(offset, 5)` onto `spans`. To decode ID 31373 later, we read its span and copy that exact slice back out — one lookup, one slice, no per-token allocation, no hashing.

`bpe/tokenizer.mojo:100` wraps the arena into something with ID-shaped semantics:

```mojo
struct TokenByteTable(ImplicitlyCopyable & Sized & Writable):
    var arena: ByteSpanArena
```

Think of `TokenByteTable` as `id -> bytes`, full stop. If the coat-check analogy for the arena clicks, decode later on will feel completely obvious — it's the same "give me the ticket, get the coat" operation.

## Stage 2: Pre-tokenization — Split Before You Merge

### Why we cannot skip this

BPE's merge step, as described above, only ever merges pairs of *adjacent* tokens. But adjacent according to what boundary? If we fed an entire sentence into the merge loop as one undifferentiated byte string, the trailing `he` of `the` and the leading `he` of `hello` in `the hello` would look identical to the algorithm, and merges could bleed across word boundaries in ways that make no linguistic sense and don't generalize.

Pre-tokenization solves this by chopping text into words and word-like pieces *before* any merging happens, using rules chosen to match what GPT-2 and GPT-4 actually do — this is a big part of what "tiktoken-compatible" means in practice, not just "same algorithm" but "same chunk boundaries."

See it directly (real output from this repo):

```python
tok2 = mbpe.GPT2Tokenizer()
tok2.train(['hello world', 'hello there'], 300)
tok2._tok.pretokenize("Hello world! Don't stop")
# [b'Hello', b' world', b'!', b' Don', b"'t", b' stop']
```

Two things worth pointing out to a newcomer here. First, notice `' world'` keeps its *leading* space as part of the token, not a separate space token — GPT-2-style tokenizers attach whitespace to the following word, which is why `hello` and ` world` are different token *identities* even though they share four letters. Second, notice `Don't` splits as `b' Don'` + `b"'t"`, not `b' Do'` + `b"n't"`. That specific split is deliberate, not accidental — it mirrors GPT-2's regex behavior exactly, because IDs have to match `tiktoken` token-for-token, and a different split boundary would silently produce different (wrong) IDs downstream.

### The code

`bpe/pretokenizer.mojo:357` defines the contract every pre-tokenizer must satisfy:

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

`PT` is a *compile-time* parameter, not a runtime flag. `BPETokenizer[GPT2Pretokenizer]` and `BPETokenizer[GPT4Pretokenizer[...]]` are genuinely different types, each with its splitting logic baked directly into the compiled binary. There is no `if model == "gpt2": ...` branch sitting in the hot loop, checked millions of times per encode call — the compiler already picked the branch for you, once, at build time. That's what "compile-time `PreTokenizer` trait" means in the README, and it's a real (if modest) performance decision, not just an architectural flourish.

There are two shipped implementations:

- **`GPT2Pretokenizer`** (`bpe/pretokenizer.mojo:457`, `name() == "gpt2"`): recognizes letter runs, digit runs, punctuation runs, the specific contractions `'t`, `'ll`, `'ve`, `'re`, and whitespace handling. `_best_match` at line 600 is the core dispatcher — it tries, in order, contraction match, then letters, then digits, then punctuation, then whitespace.
- **`GPT4Pretokenizer[mapping]`** (`bpe/pretokenizer.mojo:663`): same overall shape, different split details (for instance, how runs of digits like `123` or patterns like `.S` are handled differs from GPT-2's rules). It also takes a `mapping` parameter controlling byte shuffling — the subject of the next section.

Neither implementation reaches for a regex engine. Both are hand-written byte scanners walking a `Span[UInt8]` directly, using helpers like `_match_letter_run`, `_match_digit_run`, and `_match_punct_run`, backed by `BYTE_CLASS` lookup tables and the Unicode classification tables in `bpe/unicode_tables.mojo`. That's a deliberate trade: more code to write and maintain up front, in exchange for not paying regex-engine overhead on every single split, on every single word, for the lifetime of the tokenizer. It's also why `main.mojo` carries split-alignment tests like `test_gpt2_splits` and `test_gpt4_splits` — hand-written scanners need something locking down the exact chunk boundaries, because there's no regex source-of-truth to defer to.

One Mojo-specific detail worth flagging for newcomers: `split_view` returns `StringSlice` values, which are zero-copy *views* into the original text — no new memory allocated. `split` copies those views into owned `String` values, which does allocate. Training deliberately uses `count_words`, which feeds slices straight into a counter without ever allocating a `String`. That distinction sounds pedantic until you're training on megabytes of text and every unnecessary allocation is now happening millions of times.

## Stage 3: The Weird One — o200k Shuffles Its Bytes

### The limitation of "byte N = ID N"

For GPT-2 and cl100k, the mapping from raw byte value to starting token ID is the identity function: byte value `97` (the character `a`) simply *is* token ID 97. Nothing to look up, nothing to compute.

GPT-4o's `o200k` encoding does not do this. It permutes the mapping: rank 0 turns out to be `!` (byte 0x21), the space character (0x20) is rank 220, and `a` (0x61) is rank 64. If your code assumes "byte value = starting ID" for `o200k`, as it can safely assume for the other two encodings, every single ID it produces will be wrong — not obviously wrong, just silently, plausibly-shaped wrong, which is the worst kind of bug.

### The code

`bpe/pretokenizer.mojo:153` introduces the concept explicitly as a type, rather than leaving it as an implicit assumption baked into a function somewhere:

```mojo
struct ByteMapping(ImplicitlyCopyable & Equatable):
    var _value: Int
    comptime SEQUENTIAL = ByteMapping(0)
    comptime SHUFFLED = ByteMapping(1)
```

Alongside two compile-time lookup tables:

```mojo
comptime O200K_BYTE_TO_ID = SIMD[DType.int32, 256](...)
comptime O200K_ID_TO_BYTE = SIMD[DType.int32, 256](...)
```

`GPT4Pretokenizer[SEQUENTIAL].byte_to_id(b)` just returns `b` unchanged. `GPT4Pretokenizer[SHUFFLED].byte_to_id(b)` looks the answer up in the table instead. Because the choice is a `comptime` parameter, the `comptime if` branch selecting between these two behaviors disappears entirely at compile time — a sequential-mapping build pays literally nothing, not even a branch check, for shuffle support it will never use.

A cheat sheet, since keeping these three straight by name alone is genuinely easy to mess up:

| Python class | Mojo type | File name | Byte map |
|---|---|---|---|
| `GPT2Tokenizer` | `BPETokenizer[GPT2Pretokenizer]` | `gpt2.tiktoken` | sequential |
| `GPT4Tokenizer` | `BPETokenizer[GPT4Pretokenizer[SEQUENTIAL]]` | `cl100k.tiktoken` | sequential |
| `GPT4oTokenizer` | `BPETokenizer[GPT4Pretokenizer[SHUFFLED]]` | `o200k.tiktoken` | shuffled |

Same input string, three different (correct, by design) outputs — real output:

```python
mbpe.get_encoding("gpt2").encode("hello world")    # [31373, 995]
mbpe.get_encoding("cl100k").encode("hello world")  # [15339, 1917]
mbpe.get_encoding("o200k").encode("hello world")   # [24912, 2375]
```

If you're ever debugging an `o200k` test that fails with output that *looks* plausible but is subtly shifted from what you expected, check `ByteMapping` before you look anywhere else. It is, by a wide margin, the easiest mistake to make in this codebase, and the one I'd bet money a newcomer trips on first.

## Stage 4: Training — Counting Pairs and Merging the Winner

### The idea, with a toy example

Take the corpus `["aaabdaaabac"]` — this is literally `CANONICAL_CORPUS` in `tests/python/conftest.py`, used precisely because it's small enough to trace by hand. Start with the individual byte-tokens `a`, `b`, `c`, `d`. Count every adjacent pair: `aa` shows up 4 times, `ab` shows up twice, and so on. Merge the winner, `aa`, into a brand-new token, ID 256. Rewrite the corpus with that merge substituted in everywhere it occurred. Repeat the whole process — count pairs again in the *rewritten* corpus, merge the new winner, assign the next ID. Every merge gets the next free ID in sequence, so frequent patterns progressively collapse into single tokens.

A real (tiny) training run from this repo:

```python
tok2 = mbpe.GPT2Tokenizer()
tok2.train(['hello world', 'hello there'], 300)
tok2.encode('hello world')  # [259, 267] — small IDs because the vocab is tiny
```

With a full 50,257-entry vocabulary, `hello` collapses all the way down to a single token (31373). With a toy 300-entry vocabulary trained on two sentences, it's still represented as a few pieces stitched together — same algorithm, just stopped much earlier, with far fewer opportunities to have discovered `hello` as worth its own slot.

### The code

`bpe/tokenizer.mojo:248` is where training begins:

```mojo
def train(mut self, corpus: Span[String], vocab_size: Int) raises:
    if vocab_size < 256:
        raise Error("vocab_size must be at least 256 ...")
    var word_counts = WordCounts()
    for text in corpus:
        self.pt.count_words(text, word_counts)
```

Step one is counting words — really, pre-tokenized chunks — along with their frequencies. From there the code builds three parallel structures: an `arena` holding every word's bytes as rank IDs, plus `word_offs` / `word_len` / `word_freq` describing where each word lives and how often it occurred, plus `stats` (a pair → count map) and `where_dict` (a pair → "which words contain it" index).

The main loop, `bpe/tokenizer.mojo:318`, is the textbook BPE loop made concrete:

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

Two implementation choices are doing the real work of making this fast rather than merely correct:

1. **Incremental statistics.** A naive, textbook-literal implementation recomputes pair counts over the *entire* corpus on every single merge — that's roughly O(vocab_size × corpus_size) work in total, and it gets painfully slow past a few thousand merges. This implementation instead only revisits the words that actually contained the just-merged pair, updating only the neighboring pair counts those words touch. That's the "O(N) instead of O(V×W)" claim in the README, and it's the difference between training finishing in seconds versus not finishing in a reasonable session at all.

2. **Two-tier lookup.** A `MergeRule` (`bpe/tokenizer.mojo:19`) is a simple triple:

```mojo
struct MergeRule(TrivialRegisterPassable & Hashable & Equatable & Writable):
    var first: Int
    var second: Int
    var merged: Int
```

`MergeLookup` (`bpe/tokenizer.mojo:63`) stores these in a flat array for small IDs, falling back to a `Dict` only for the rest:

```mojo
struct MergeLookup(ImplicitlyCopyable & Writable):
    var _fast: IntArray   # CACHE_ENTRIES flat array
    var _slow: Dict[Int, Int]
```

`get(id1, id2)` checks the flat array directly when both IDs are under 1000, and only falls through to the (slower, hash-based) dict otherwise. Encode performs millions of these lookups over the life of a program, so keeping the *common* case — small, frequently-seen IDs — off the hashing path is a meaningful win, not a premature-optimization footnote.

## Stage 5: Encode — From Text to IDs

### The flow

```
text -> split into words (PreTokenizer)
     -> bytes -> ranks (byte_to_id)
     -> apply merges lowest-rank-first until stuck
     -> append IDs to result
```

Special tokens, like `<|endoftext|>`, are handled one level above this — covered separately below, because they interact with encoding in a way that's worth isolating.

### The code

`encode_ordinary` (`bpe/tokenizer.mojo:530`) is the hot path exercised when no special tokens are in play. For each pre-tokenized word, it picks between two different merge strategies based on the word's length:

```mojo
if n < 2:
    self._copy_word_ids(ptr, n, dst)
elif n < SCAN_LIMIT:   # 32
    write_pos += self._merge_scan(ptr, n, dst)
else:
    write_pos += self._merge_heap(ptr, n, dst, scratch, heap)
```

`_merge_scan` is deliberately simple: repeatedly find the lowest-ranked mergeable pair anywhere in the word and merge every occurrence of it (`merge_inplace`). That's technically O(n²) in the word length, which sounds alarming until you remember that almost all real words are short — n is nearly always under 10 or so — so the constant-factor simplicity wins outright over anything fancier at that scale.

`_merge_heap` exists for the rare long word, using a `BinaryHeap` of candidate merges keyed by rank, backed by a linked-list representation (`ids` / `nxt` / `prv` / `alive` fields inside `MergeScratch`) so that each merge is an O(log n) heap update rather than a full rescan of the word. Short words use the simple scanner; long words use the heap. Same final answer either way — this is purely a speed decision, invisible from the outside.

`_copy_word_ids` is where the `SHUFFLED` byte mapping from Stage 3 actually gets consulted:

```mojo
comptime if Self.PT.byte_map == ByteMapping.SHUFFLED:
    dst[i] = Self.PT.byte_to_id(Int(ptr[i]))
else:
    dst[i] = btr[Int(ptr[i])]
```

For sequential mappings, it reads directly from a per-tokenizer `byte_to_rank` array, rebuilt whenever a tokenizer is loaded. For shuffled mappings, it calls into the compile-time lookup table instead. Same `comptime if` trick as before — one build path or the other is compiled in, never both, never a runtime check.

`encode` itself (`bpe/tokenizer.mojo:586`) wraps `encode_ordinary` with special-token scanning layered on top. It walks the input text, checks each position against the set of registered `special_bytes`, emits the special's ID directly whenever one matches, and encodes everything in between as ordinary text. When `special_bytes` is empty, it skips straight through to `encode_ordinary` with effectively zero overhead — a path explicitly locked down by `test_special_tokens_no_specials_registered` in `main.mojo`, so that supporting specials never taxes the (far more common) case of not using any.

## Stage 6: Decode — From IDs Back to Text

Decode is the easy direction, and it's worth appreciating just how much easier: look up each ID's bytes, concatenate them, done. No merges to replay, no pre-tokenizer involved, no heap.

`bpe/tokenizer.mojo:657`:

```mojo
def decode(self, ids: Span[Int]) raises -> String:
    # 1. validate range, sum total length
    # 2. String(unsafe_uninit_length=total)
    # 3. unsafe_memcpy each token's bytes into place
```

That asymmetry — decode is a pure `memcpy` chain, encode has to run a whole merge search — is exactly why decode throughput in the README runs roughly 10× encode throughput (184.5M tok/s versus 17.7M tok/s for GPT-2, native Mojo). It's not a tuning accident; it's structural. Decode simply has less work to do by the nature of the problem.

A few siblings worth knowing exist alongside `decode`:

- `decode_bytes` — identical, but returns a raw `ByteArray` instead of a `String`.
- `decode_single_token_bytes(id)` — the bytes for exactly one token, used by merge-consistency checks (see Stage 7).
- `decode_with_offsets(ids, starts, ends)` — returns the decoded text *plus* per-token `(start, end)` byte offsets into it. The Python wrapper turns those into a list of tuples, useful for anything that needs to map model output back to spans in the original text (highlighting, citation, etc.).
- `display_of(id)` / `encode_single_token(text)` — debugging helpers that map through printable characters, handy when you're staring at a wrong ID and need to see what it actually represents.

All of them raise `Error("token ID out of range")` on invalid IDs, covered by `tests/python/test_error_handling.py`.

## Stage 7: Files — The `.tiktoken` Format and Merge Recovery

### What is actually on disk

A `.tiktoken` file is nothing exotic: plain text lines of `base64(bytes) rank`.

```
IQ== 0
Ig== 1
Iw== 2
...
```

The first lines decode to single bytes. Later lines are multi-byte merge products — the same tokens the training loop discovered, in the same order it discovered them. Real counts from this repo's data files:

```
50256 data/gpt2.tiktoken
100256 data/cl100k.tiktoken
199998 data/o200k.tiktoken
```

`data/` also ships `p50k_base.tiktoken`, `p50k_edit.tiktoken`, and `o200k_harmony.tiktoken` for older/alternate encodings.

### Save

`save_tiktoken` (`bpe/tokenizer.mojo:946`) walks the token table, skips anything registered as a special token, base64-encodes each token's bytes, and writes `encoded + " " + id` per line. Two consequences follow directly from this:

1. Saves are deterministic — the same tokenizer state always produces a bit-identical file (verified by `test_tiktoken_deterministic_save`), which matters if you ever need to diff two training runs or check a file into version control meaningfully.
2. Special tokens are **not** persisted to the file at all. This is the single most common thing that trips people up, mentioned again below.

### Load and merge recovery

`load_tiktoken` (`bpe/tokenizer.mojo:968`) reads the file back, rebuilds `mergeable_ranks` (bytes → rank) and `all_tokens` (rank → bytes), reconstructs the token table, and then calls `_recover_merges`.

Why is recovery even necessary? Because the `.tiktoken` file only records *what* each token is and its rank — it does not record the merge history (which two earlier tokens combined to produce this one). Encoding, though, needs that merge history in priority order to work at all. `_recover_merges` reconstructs it by replaying BPE for every token: for each token ID at or above 256, it reruns the merge search restricted to ranks below that ID, finds which two pieces merged last to produce it, and records that as a `MergeRule(left, right, id)`. `main.mojo:test_tiktoken_merge_consistency` checks the resulting invariant holds for *every* recovered merge:

```
bytes[merged] == bytes[left] + bytes[right]
```

After recovery, `byte_to_rank` is rebuilt and any specials defined by the pre-tokenizer type itself are auto-registered:

```mojo
for item in Self.PT.special_tokens().items():
    if not item.value in self.inverse_special:
        self._register_special_token(item.key, item.value)
```

So loading `gpt2.tiktoken` into a `BPETokenizer[GPT2Pretokenizer]` always ends up with `<|endoftext|>` mapped to 50256, even though the file on disk never mentions it explicitly — that registration comes from the pre-tokenizer type, not the file. Any *custom* special tokens you added yourself must be re-registered by hand after loading; see `test_special_tokens_save_load` in `main.mojo:704` for the pattern.

`Tokenizers.get` (`bpe/tokenizer.mojo:1059`) ties the whole load path together into one call:

```mojo
struct Tokenizers:
    comptime gpt2 = GPT2Pretokenizer
    comptime cl100k = GPT4Pretokenizer[ByteMapping.SEQUENTIAL]
    comptime o200k = GPT4Pretokenizer[ByteMapping.SHUFFLED]

    @staticmethod
    def get[T: PreTokenizer](filename: String = "") raises -> BPETokenizer[T]:
        var fname = T.name() if filename.byte_length() == 0 else filename
        var tok = BPETokenizer[T]()
        tok.load_tiktoken(_find_data_dir() + "/" + fname + ".tiktoken")
        return tok^
```

`_find_data_dir` checks the `MBPE_DATA_DIR` environment variable first, falling back to `./data`. If you're running Mojo natively and your data lives somewhere else, set that variable. The Python path resolves data relative to the installed package instead — a distinction that bites people, covered in the pitfalls section below.

## Stage 8: Python Bindings — Mojo Speed, tiktoken API

### The two layers

Most people who use this library will never touch a line of Mojo. They'll do this:

```python
import mbpe
tok = mbpe.get_encoding("gpt2")
tok.encode("hello world", allowed_special={"<|endoftext|>"})
```

That single call passes through two distinct layers underneath.

**Layer one: `python-binding/mbpe.mojo`**, which compiles down to a `_mbpe.so` shared library. It defines three concrete, fully-specialized types:

```mojo
comptime GPT2TK = BPETokenizer[GPT2Pretokenizer]
comptime GPT4TK = BPETokenizer[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]
comptime GPT4oTK = BPETokenizer[GPT4Pretokenizer[ByteMapping.SHUFFLED]]
```

Each type is exposed to Python with `train`, `encode`, `encode_ordinary`, `decode`, `save/load_tiktoken`, `register_special_tokens`, `decode_bytes`, `decode_with_offsets`, and more, via `PythonModuleBuilder`. Methods are duplicated per concrete type — `_encode_gpt2`, `_encode_gpt4`, `_encode_gpt4o`, and so on — as a deliberate workaround for a Mojo trait-downcast limitation noted directly in a comment at line 80. It's not elegant, but it's honest about why it exists rather than hiding the limitation.

Build it with:

```bash
pixi run mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/mbpe/_mbpe.so
```

The resulting file is gitignored — it's a build artifact, not source. `scripts/run_tests.sh` deletes and rebuilds it on every run specifically so tests can never silently run against a stale binary from an earlier edit.

**Layer two: `python-binding/mbpe/__init__.py`**, which wraps `_mbpe` to match `tiktoken`'s actual Python API surface. The Mojo layer only understands `encode` (special-token-aware) versus `encode_ordinary` (specials ignored, treated as plain text). All of `tiktoken`'s richer `encode(allowed_special=..., disallowed_special=...)` semantics live entirely in Python, in `_BaseTokenizer.encode` (lines 29–121). That function normalizes `allowed_special` (accepting the string `"all"`, a set, or nothing), enforces `disallowed_special` (`"raise"` versus `"ignore"`), and for the partial-subset case, manually splits the input on the allowed specials and calls `encode_ordinary` on each gap in between.

`get_encoding` maps encoding names to the right class and loads the matching file. Note that `r50k_base` actually loads `gpt2.tiktoken` under the hood, via `_ENCODING_FILE` — a naming quirk inherited from `tiktoken` itself, preserved here for compatibility. The `encode` vs. `encode_ordinary` distinction shows up clearly in real output:

```python
tok.encode("<|endoftext|>hello world")      # [50256, 31373, 995]
tok.encode_ordinary("<|endoftext|>hello")   # [27, 91, 437, 1659, 5239, 91, 29, 31373]
```

`encode` recognizes the special marker and emits it as a single clean ID. `encode_ordinary` has no concept of "special" at all and shreds the same text, byte by byte and merge by merge, into eight separate tokens. That's the single most common point of confusion for people new to this codebase (or to `tiktoken` generally), and the test suite leans on it hard — see `test_encode_params.py` and `test_parity_special_tokens.py`.

## The Demo: Put It All Together

Here is the smallest script that touches every stage above in order. Every output shown below is real, copied directly from commands run against this repo while writing this post — nothing here is a hypothetical or a "should print."

```python
import sys
sys.path.insert(0, 'python-binding')
import mbpe

# Load (Stage 7 + Stage 8)
tok = mbpe.get_encoding("gpt2")
print(tok.encode("hello world"))      # [31373, 995]
print(tok.decode([31373, 995]))       # 'hello world'
print(tok.n_vocab)                    # 50257

# Train tiny (Stage 4)
tiny = mbpe.GPT2Tokenizer()
tiny.train(["hello world", "hello there"], 300)
print(tiny.encode("hello world"))     # [259, 267]

# Pre-tokenize (Stage 2)
print(tiny._tok.pretokenize("Hello world! Don't stop"))
# [b'Hello', b' world', b'!', b' Don', b"'t", b' stop']

# Other encodings (Stage 3)
print(mbpe.get_encoding("cl100k").encode("hello world"))  # [15339, 1917]
print(mbpe.get_encoding("o200k").encode("hello world"))   # [24912, 2375]
```

What this proves, concretely: the same input string produces genuinely different ID sequences under each encoding, because each has a different pre-tokenizer, different learned merges, and (for `o200k`) a different byte map — yet every single one decodes back to the exact original text. Training a fresh tokenizer from scratch takes two lines of Python. The tiny vocabulary above uses small IDs simply because it only had 44 merges to learn (300 − 256 = 44) from two short sentences.

The Mojo-native equivalent, for anyone not going through Python at all:

```mojo
from bpe.tokenizer import Tokenizers

def main() raises:
    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    var ids = gpt2.encode("hello world")
    print(gpt2.decode(ids))  # hello world
```

`Tokenizers.gpt2` is just a compile-time alias for `GPT2Pretokenizer`. Swap in `Tokenizers.o200k` instead, and every other line of code stays identical — no branches added, nothing else to change. That is the entire payoff of making `PT` a compile-time parameter rather than a runtime string: swapping encodings is a one-token edit, not a refactor.

## Common Pitfalls (Read Before You Debug)

1. **Stale `_mbpe.so`.** Python imports the *compiled* shared library, not your edited `.mojo` source directly. If you change Mojo code and Python's behavior stubbornly does not change, you almost certainly forgot to rebuild. `bash scripts/run_tests.sh` always rebuilds first, specifically to prevent this.

2. **Wrong `ByteMapping` for o200k.** Using `SEQUENTIAL` against an `o200k.tiktoken` file loads without any error at all — it just silently produces wrong IDs. If `o200k` output looks "almost right but shifted," check the `SHUFFLED` parameter in `python-binding/mbpe.mojo:21` and `bpe/tokenizer.mojo:1063` before you look anywhere else.

3. **Specials vanish across save/load.** `save_tiktoken` skips them by design, as covered in Stage 7. After `load_tiktoken`, only the specials defined by `PT.special_tokens()` come back automatically. Custom specials you registered yourself need to be re-registered by hand after every reload — see `main.mojo:704`.

4. **`encode` vs. `encode_ordinary` confusion.** If `<|endoftext|>` comes back as 8 separate tokens instead of the single ID `[50256]`, you either called `encode_ordinary` by mistake, or called `encode` with `allowed_special=set()` and `disallowed_special="ignore"`. That is correct, intentional behavior in both cases — not a bug to file.

5. **`MBPE_DATA_DIR` confusion.** Mojo's `Tokenizers.get` looks at `$MBPE_DATA_DIR`, falling back to `./data/`. Python's `get_encoding` instead resolves data relative to the installed package. Setting the environment variable fixes the Mojo path and does precisely nothing for the Python path, and vice versa — the two loaders don't share configuration.

6. **Forgetting `-I .`.** The `bpe/` package only resolves correctly with `mojo ... -I .`, with the single exception of `main.mojo`, which is run as `pixi run mojo main.mojo` without that flag. When in doubt, copy the exact invocation from `scripts/run_tests.sh` rather than reconstructing it from memory.

## Where to Go Next in the Repo

- **Want the split rules in full?** Read `bpe/pretokenizer.mojo:624` (`GPT2 split_view`) alongside the GPT-4 version at line 1141. Then run `pixi run mojo main.mojo` and deliberately break a test to watch the failure mode — that tells you more than reading in isolation.
- **Want merge speed detail?** Read `_merge_scan` versus `_merge_heap` in `bpe/tokenizer.mojo:438` and `bpe/tokenizer.mojo:463` side by side. Benchmark it yourself with `pixi run benchmark` — fair warning, it's slow, since it regenerates corpora and pulls in the Rust toolchain for comparison.
- **Want proof of parity, not just a claim of it?** Read `tests/python/test_parity_ids.py` and `tests/python/parity_helpers.py`. Run `pixi run --environment dev python -m pytest tests/python/ -k gpt2 -v` and watch it pass against real `tiktoken` output.
- **Want the file format in your own hands?** Open `data/gpt2.tiktoken` and decode the first few lines yourself: `IQ==` is byte 0, `Ig==` is byte 2. Then read `save_tiktoken` / `load_tiktoken` in `bpe/tokenizer.mojo:946`.
- **Want to add a new encoding?** Write a new `PreTokenizer` struct, implement its `special_tokens()` and `name()`, expose it in `Tokenizers` and in `python-binding/mbpe.mojo:py_get_encoding`, add a `.tiktoken` data file, and write a parity test against the reference implementation. That's the whole checklist — nothing hidden elsewhere.

## Takeaways

We started from a simple, almost embarrassing fact — models see integers, not text — and ended up with three interchangeable, from-scratch tokenizers sharing one engine underneath. The pre-tokenizer decides where chunk boundaries fall. Training counts adjacent pairs and merges the winners into `MergeRule`s, cached for fast lookup in a `MergeLookup`. Encoding applies those learned merges per word using a scan-or-heap strategy chosen by word length. Decoding is, by contrast, nothing more than a `memcpy` over an arena — genuinely the easy direction. Files on disk store base64-encoded bytes plus ranks, and the merge history that encoding needs is reconstructed on load rather than stored redundantly. Python adds `allowed_special` handling on top of a thin, compiled Mojo shared library.

None of this is presented as the fastest or most polished tokenizer available — there are faster, more battle-tested implementations, and I'd rather say that plainly than let the numbers imply otherwise. What I'm confident stating is narrower and, I think, more useful: every stage above is one I built, tested against real `tiktoken` output, and can explain to you line by line, including where it's slow and why. If you're trying to actually understand BPE rather than just call a library that does it, that kind of walkthrough is worth more than a faster black box.

Your concrete next step, if you want it to stick: train a 300-vocab tokenizer on two sentences of your own, save it with `save_tiktoken`, reload it, and verify `encode` gives you identical IDs before and after. That single loop touches every stage covered in this post. Once it feels routine, you're at home in this codebase.

---

*Further reading: Sennrich et al. (2016), "Neural Machine Translation of Rare Words with Subword Units" (ACL 2016), for the original BPE-for-NLP adaptation; the Hugging Face LLM Course, Chapter 6.5, for a shorter BPE walkthrough from a different angle; `README.md` and `CONTRIBUTING.md` in the [mbpe repository](https://github.com/ratulb/mbpe) for exact build and test commands.*
