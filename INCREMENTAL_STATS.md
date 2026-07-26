# Incremental Pair Statistics — Training Speed

## What is the problem?

Every BPE training iteration finds the most frequent adjacent token pair in the
corpus and merges it into a single new token. To find the most frequent pair,
you need pair-frequency statistics.

Our current training loop recomputes **all** pair frequencies from scratch on
every merge iteration:

```mojo
while len(self.vocab) < vocab_size:
    var pair_freqs = _compute_pair_freqs(splits, word_freqs)  # ← rescan everything
    # pick best pair, apply merge...
```

This is **O(V × W)** per merge where:

| Symbol | Meaning | Gutenberg corpus |
|---|---|---|
| V | Number of unique words | 179K |
| W | Average tokens per word | ~5 |
| M | Number of merges | 244 (vocab 500 − 256 base) |

Total token scans: **V × W × M ≈ 179K × 5 × 244 ≈ 218 million**.

Each rescan is wasted work because most pairs didn't change — only the
few that involved the just-merged token need updating.

---

## The insight: most pairs don't change

When we merge `(a, b) → new_id` at position `i` in the token sequence:

```
Before:  [ ... , prev,  a,  b,  next, ... ]
                    ↑    ↑   ↑    ↑
After:   [ ... , prev, new_id,  next, ... ]
                    ↑    ↑       ↑
```

Only **5 pairs** are affected per merge occurrence:

| Pair | Action |
|---|---|
| `(prev, a)` | ❌ Destroyed — decrement count |
| `(a, b)` | ❌ Destroyed — decrement count |
| `(b, next)` | ❌ Destroyed — decrement count |
| `(prev, new_id)` | ✅ Created — increment count |
| `(new_id, next)` | ✅ Created — increment count |

If `(a, b)` occurs 1000 times across the corpus, we do **5000 pair updates**
(1000 × 5) instead of rescaming **all 900K tokens**. That is the core savings.

---

## Data structure: flat sequence with SEP sentinel

Instead of storing each word's token IDs in a separate `List[Int]`:

```mojo
# Current: per-word dict
splits = {
    "the": [30, 5, 12],
    "cat": [5, 1, 27],
    "the": [30, 5, 12],   ← duplicate entries
    ...
}
```

We store one flat `List[Int]` with `SEP = -1` between words.
Each word appears `freq` times in sequence (frequency weighting):

```mojo
# Incremental: flat list with SEP sentinel
ids = [
    30, 5, 12,  -1,       # "the" × 1
    5, 1, 27,   -1,       # "cat" × 1
    30, 5, 12,  -1,       # "the" × 1 (second occurrence)
    ...
]
```

**Why SEP = -1?** Token IDs are never negative (0–255 for base bytes, 256+
for merged tokens). So `-1` is an unambiguous sentinel that can never collide
with a real token.

**SEP rules:**
1. A pair crossing SEP is never eligible for merging — words are atomic units
2. SEP itself is never part of a merge — it is not a valid token ID
3. Pairs that include SEP are tracked in stats but ignored when finding the
   maximum pair
4. During merge application, SEP tokens pass through unchanged

---

## The pair-frequency map

Instead of `Dict[Tuple[Int, Int], Int]` (heap-allocated Tuple per entry), we
use `Dict[Int, Int]` with a **packed key**:

```mojo
comptime ENCODE_SHIFT: Int = 20
comptime ENCODE_MASK: Int = (1 << 20) - 1

@always_inline
def _pack(first: Int, second: Int) -> Int:
    return (first << ENCODE_SHIFT) | second
```

| Operation | Code | Cost |
|---|---|---|
| Increment pair count | `stats[_pack(a, b)] = stats.get(_pack(a, b), 0) + delta` | 1 hash lookup |
| Check existence | `_pack(a, b) in stats` | 1 hash lookup |
| Find max pair | scan all entries, track `best_val` | O(unique pairs) |
| Decrement (with zero-out) | `if nc <= 0: stats[_pack(a,b)] = 0` | 1–2 lookups |

---

## Training algorithm

### Phase 1: Build the flat sequence

```
for each line of input text:
    words = PT.split(line)
    for each word:
        ids.append(SEP)
        for each byte of word:
            ids.append(byte_value)
```

The SEP at the start of each word means the list starts with a SEP, which
conveniently eliminates the "prev is undefined" edge case at position 0.

### Phase 2: Count initial pair frequencies (one pass)

```
stats = {}        # Dict[Int, Int]
for i in range(len(ids) - 1):
    if ids[i] != SEP and ids[i+1] != SEP:
        freq = word_frequency_of_the_word_starting_at(i)  ← see note
        key = _pack(ids[i], ids[i+1])
        stats[key] = stats.get(key, 0) + freq
```

**Note on frequency weighting:** In the flat-sequence approach, each word
appears `freq` times in `ids`. So every occurrence of a pair corresponds
to one actual occurrence in the corpus. No explicit weight multiplication
is needed because the repetition *is* the weight. Therefore the simple
count (increment by 1) is correct.

However, we need to distinguish between "the pair has count 0" and "the pair
does not exist in the map". We use zero as "exists but count is 0" and
absence from the dict as "never existed". This matters for the decrement
logic — we must not decrement a nonexistent pair.

### Phase 3: Merge loop

```
while len(merges) < num_merges:
    # ── 3a. Find max-frequency pair ──
    best_pair = None
    best_freq = 0
    for key, freq in stats.items():
        if freq > best_freq:
            a = key >> ENCODE_SHIFT
            b = key & ENCODE_MASK
            if a != SEP and b != SEP:
                best_freq = freq
                best_pair = (a, b)

    if best_pair is None:
        break   # no more pairs to merge

    (a, b) = best_pair
    new_id = 256 + len(merges)

    # ── 3b. Single scan: merge + update stats incrementally ──
    new_ids = List[Int](capacity=len(ids))
    i = 0
    while i < len(ids):
        if ids[i] == SEP:
            new_ids.append(SEP)
            i += 1
        elif i < len(ids) - 1 and ids[i] == a and ids[i+1] == b:
            # ──── Decrement destroyed pairs ────
            # (prev, a):
            if len(new_ids) > 0 and new_ids[-1] != SEP:
                decr(stats, new_ids[-1], ids[i])
            # (a, b):
            decr(stats, ids[i], ids[i+1])
            # (b, next):
            if i + 2 < len(ids) and ids[i+2] != SEP:
                decr(stats, ids[i+1], ids[i+2])

            # ──── Increment created pairs ────
            # (prev, new_id):
            if len(new_ids) > 0 and new_ids[-1] != SEP:
                incr(stats, new_ids[-1], new_id)
            # (new_id, next):
            if i + 2 < len(ids) and ids[i+2] != SEP:
                incr(stats, new_id, ids[i+2])

            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    ids = new_ids^

    # ── 3c. Record merge ──
    merges.append(MergeRule(a, b, new_id))
    merge_cache.set(a, b, new_id)

    # ── 3d. Build vocab entry ──
    left = vocab[a]
    right = vocab[b]
    merged_str = left + right
    vocab.append(merged_str)
    # ... build token_bytes with Ġ→0x20 substitution ...
```

### Helper: decrement-with-zero-out

```mojo
@always_inline
def decr(mut stats: Dict[Int, Int], a: Int, b: Int):
    var key = _pack(a, b)
    if key in stats:
        var new_val = stats[key] - 1
        if new_val <= 0:
            stats[key] = 0
        else:
            stats[key] = new_val

@always_inline
def incr(mut stats: Dict[Int, Int], a: Int, b: Int):
    var key = _pack(a, b)
    stats[key] = stats.get(key, 0) + 1
```

The guard `if key in stats` prevents creating entries for `(prev, a)` when
that pair never existed (e.g., because `a` was just created in the previous
merge and `prev` is a token that never appears before `a`).

---

## Full worked example

**Corpus:** `"aa"` occurs 3 times, `"ab"` occurs 2 times.

### Phase 1: Build flat sequence

Pre-tokenization (assuming G convention splits on spaces):

```
"aa" → ["aa"]   (each word is one split)
"ab" → ["ab"]
```

Flat sequence (repeated by frequency):

```
ids = [a, a, SEP,  a, a, SEP,  a, a, SEP,  a, b, SEP,  a, b]
        ↑--- "aa" × 3 ---↑        ↑--- "ab" × 2 ---↑
```

### Phase 2: Initial pair counts

```
stats:
  (a,a): 3   (from the three "aa" words)
  (a,b): 2   (from the two "ab" words)
  (SEP,a): 5   (from every word start)
  (a,SEP): 3   (from every "aa" word end)
  (b,SEP): 2   (from every "ab" word end)
```

SEP-crossing pairs (SEP,a) and (a,SEP) and (b,SEP) are tracked but ignored
when picking `best_pair`.

### Phase 3: Merge iterations

```
MERGE 1: best_pair = (a,a) freq=3, new_id=256

Scan ids, build new_ids, update stats:
  pos=0: ids[0]=a, ids[1]=a → MATCH
    decr(prev,none) — skip (no prev)
    decr(a,a)       stats[(a,a)]: 3→2
    decr(b,none)    — skip (i+2=2 is SEP)
    incr(prev,none) — skip
    incr(new,next)  — skip (next is SEP)
    new_ids.append(256), i+=2

  pos=1: ids[1]=SEP → pass through
    new_ids.append(SEP), i+=1

  pos=2: ids[2]=a, ids[3]=a → MATCH
    decr(prev,SEP)  — skip (prev is SEP)
    decr(a,a)       stats[(a,a)]: 2→1
    decr(b,none)    — skip (i+2=4 is SEP)
    incr(prev,none) — skip (prev is SEP)
    incr(new,next)  — skip (next is SEP)
    new_ids.append(256), i+=2

  ... (third "aa" same pattern)

  pos=6: ids[6]=a, ids[7]=b → NO MATCH (b ≠ a)
    new_ids.append(a), i+=1

  pos=7: ids[7]=b → NO MATCH
    new_ids.append(b), i+=1

  pos=8: ids[8]=SEP → pass through
  pos=9: ids[9]=a, ids[10]=b → NO MATCH

ids after merge 1:
  [256, SEP, 256, SEP, 256, SEP, a, b, SEP, a, b]

stats after merge 1:
  (a,a): 1      (one "aa" word still unmerged — but wait, we merged all three)

Hmm, let me re-examine. After merge 1, all three "aa" words should be fully
merged to [256]. So (a,a) should be 0. Let me trace more carefully...

Actually, I see the issue. When we merge at positions 0,1 and then advance
i by 2, the next element at the old index 2 is SEP, which passes through.
So the three "aa" words do get fully merged. stats[(a,a)] should be 0 after
merge 1.

But there's a subtlety: we only decrement (a,a) when we actually encounter
a match. The three "aa" words each have one (a,a) pair, and all three get
merged, so (a,a) goes 3→2→1→0. Correct.

MERGE 2: best_pair = (a,b) freq=2, new_id=257

  pos in old ids: [256, SEP, 256, SEP, 256, SEP, a, b, SEP, a, b]
  Scan proceeds, finds (a,b) at positions 7 and 10.
  Each match decrements (prev,a), (a,b), (b,next) and increments
  (prev,new), (new,next).

  After merge 2:
    ids = [256, SEP, 256, SEP, 256, SEP, 257, SEP, 257]
    stats: (a,b)→0, (257,SEP)→2, (a,SEP)→1→0, ...
```

Result: 3 tokens [256] (from "aa") and 2 tokens [257] (from "ab").
Same as the original algorithm.

---

## What remains unchanged

| Part of training | Change |
|---|---|
| Pre-tokenization (`PT.split()`) | Same — still produces word strings |
| Byte-to-codepoint mapping | Same — 256 base bytes |
| Base vocab initialization | Same — IDs 0–255 |
| Vocab display string building | Same — `vocab[a] + vocab[b]` |
| Ġ→0x20 substitution in token_bytes | Same |
| Merge rule recording | Same — `merges.append(MergeRule(a,b,merged_id))` |
| PairCache population | Same — `merge_cache.set(a,b,merged_id)` |
| Encoding (`_tokenize()`, `encode()`) | Same — uses PairCache, not stats |
| Decode | Same |
| Save/Load | Same |

| Part of training | Change |
|---|---|
| `splits: Dict[String, List[Int]]` | ❌ Removed — replaced by flat `ids: List[Int]` |
| `word_freqs: Dict[String, Int]` | Still used for pre-tokenization phase, not for pair counting |
| `_compute_pair_freqs()` | ❌ Removed — no longer called |
| `_merge_pair()` | ❌ Removed — merge is inlined in the scan |
| Pair counting | Fresh `Dict[Int,Int]` per merge → persistent `Dict[Int,Int]`, incrementally updated |

---

## Performance analysis

### Per-merge cost

| Operation | Old (rescan) | New (incremental) |
|---|---|---|
| Scan all word lists | V × W ≈ 900K | — |
| Build pair_freqs Dict | V × W × hash ops ≈ 900K | — |
| Find max pair | O(unique pairs) × Dict scan | O(unique pairs) × Dict scan (same) |
| Apply merge + update stats | V × W scans + list rebuild | 1 flat scan + 5 ops per occurrence |
| Dict overhead | Fresh allocation each merge | Zero — map persists |

### Total training time (estimated)

For Gutenberg corpus (900K tokens, 244 merges):

| | Old | New | Speedup |
|---|---|---|---|
| Pair counting | 244 × 900K = 220M ops | 900K (one pass) | ~244× |
| Merge application | 244 × 900K = 220M ops | 244 × 900K = 220M ops | Same |
| Dict memory churn | 244 allocations | 1 allocation | ~244× |
| **Total estimate** | ~600 ms | ~300 ms (unsure — dominated by merge scan) | |

The pair-counting elimination saves ~50% of training time on this corpus.
On larger corpora (millions of tokens) the savings grow proportionally.

### Scaling

| Corpus size | Old training time | New training time |
|---|---|---|
| 1 MB (Gutenberg) | ~600 ms | ~300 ms |
| 10 MB | ~6 s | ~3 s |
| 100 MB | ~60 s | ~30 s |
| 1 GB | ~10 min | ~5 min |

The constant-factor improvement is ≈2×. The asymptotic improvement is that
old time scales with V × W × M while new time scales with total_tokens × M
(the merge scan dominates both, but the pair-count rescans are eliminated).

---

## Edge cases

### Empty corpus
If the corpus has no tokens, `ids` is empty and the merge loop never runs.
This is handled by checking `len(ids) == 0` before the merge loop.

### Single-token words
A word that becomes a single token has no internal pairs. In the flat
sequence it looks like `[token, SEP]` — the only pair is `(token, SEP)`,
which is ignored because SEP is involved. No merge can occur within it.

### Words that share no pairs
The stats map may become empty (all pairs have merged into single-token
words). The `best_pair` check catches this and breaks out of the loop.

### Very frequent merges
If a pair occurs 50,000 times, the incremental update does 50,000 × 5 =
250,000 Dict operations. This is still faster than rescaming all 900K tokens
and rebuilding the entire pair-freq dict.

---

## Relationship to bpe.mojo

The algorithm described here is a port of `bpe.mojo/bpe/basic_tokenizer.mojo`
(lines 114–230) adapted for our `BPETokenizer[PT]` architecture. Key differences:

| Aspect | bpe.mojo | Our adaptation |
|---|---|---|
| Pretokenization | None (raw bytes) | Via `PT.split()` |
| Sentinels | No sentinels (single flat sequence) | SEP = -1 between words |
| Weighting | Unweighted (each byte = 1 occurrence) | Frequency-weighted (words repeated by freq) |
| Encoder | Separate merge-map lookup | Our existing `PairCache` |
| Vocab storage | `FlatTokenStorage` (UnsafePointer) | Our inline `token_bytes`/`offsets`/`lengths` |
