# Test Implementation Plan — `tokenizer_tests.txt`

## Scope

`tests/tokenizer_tests.txt` defines ~110 test stubs across 16 sections.
We already have **78 passing tests** (36 `main.mojo` + 9 `tests/test_tokenizer.mojo` + 33 `tests/exhaustive_tokenizer.mojo`).

Below: each section categorized and prioritized.

---

## Legend

| Icon | Meaning |
|------|---------|
| ✅ | Already covered by existing 45 tests |
| 🟢 | Feasible with current code — new test only |
| 🟡 | Needs minor source change (validation, error-handling) |
| 🔴 | Internal detail / not applicable / not feasible in Mojo |
| ❌ | Skip (no Mojo equivalent: None, dynamic types, property fuzzing) |

---

## Section 0 — INITIALIZATION & CONFIGURATION

| # | Test | Status | Notes |
|---|------|--------|-------|
| 0.1 | `test_initialization_default_params` | 🔴 | No custom init params in our API |
| 0.2 | `test_initialization_custom_params` | 🔴 | No custom params pattern |
| 0.3 | `test_initialization_invalid_params_raises_valueerror` | 🔴 | N/A |
| 0.4 | `test_vocab_size_below_byte_range_raises_valueerror` | 🟡 | Currently trains 0 merges silently. Could validate. |
| 0.5 | `test_pat_str_customization_affects_tokenization` | 🔴 | No pat_str |
| 0.6 | `test_invalid_pattern_raises_valueerror` | 🔴 | No pattern |

**Decision: Skip section 0** — not applicable to our architecture.

---

## Section 1 — VOCABULARY & TOKEN MANAGEMENT

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1.1 | `test_vocab_initial_state_is_256_bytes` | ✅ | Implicitly covered (byte-level, roundtrip tests) |
| 1.2 | `test_vocab_duplicate_token_raises_valueerror` | 🔴 | No `add_token` API |
| 1.3 | `test_vocab_ids_are_stable` | ✅ | Covered by `test_deterministic` + tiktoken determinism |
| 1.4 | `test_vocab_unique_ids` | 🟢 | Verify no two vocab entries are equal strings |
| 1.5 | `test_special_tokens_not_reordered` | ✅ | Covered by special token mapping tests |
| 1.6 | `test_vocab_size_matches_expected` | 🟢 | After train(corpus, N), verify len(tok) == min(N, 256+merges) |

**New tests: 2** (1.4, 1.6)

---

## Section 2 — RANK TABLE / PAIR-LOOKUP CACHE

| # | Test | Status | Notes |
|---|------|--------|-------|
| 2.1–2.9 | All PairCache-internal tests | 🔴 | Internal implementation detail. Not exposed. |
| 2.10 | `test_rank_table_survives_save_load_roundtrip` | ✅ | Covered by tiktoken roundtrip tests |
| 2.11 | `test_rank_table_lookup_is_O1_not_On` | 🔴 | Can't measure in CI |

**Decision: Skip section 2** — PairCache is an internal detail.

---

## Section 3 — MERGE RULES

| # | Test | Status | Notes |
|---|------|--------|-------|
| 3.1 | `test_single_merge_applies_correctly` | ✅ | Covered by Wikipedia example (`test_wikipedia_example`) |
| 3.2 | `test_multiple_sequential_merges` | ✅ | Same |
| 3.3 | `test_recursive_merges_chain_correctly` | ✅ | Same |
| 3.4 | `test_no_merge_possible_returns_input_unchanged` | 🟢 | Encode text with no mergeable pairs → byte-level output |
| 3.5 | `test_longest_merge_chain_reaches_expected_token` | ✅ | Wikipedia example covers this |
| 3.6 | `test_duplicate_merge_rank_raises_valueerror` | 🔴 | No direct merge table API |
| 3.7 | `test_duplicate_merge_rule_raises_valueerror` | 🔴 | Same |
| 3.8 | `test_circular_merge_detection_raises_valueerror` | 🟡 | Would need merge graph cycle detection |

**New tests: 1 + 1 optional** (3.4 🟢, 3.8 🟡)

---

## Section 4 — TRAINING

| # | Test | Status | Notes |
|---|------|--------|-------|
| 4.1 | `test_train_on_single_text` | ✅ | Covered by `test_basic_roundtrip` |
| 4.2 | `test_train_on_multiple_texts` | ✅ | Covered by `test_full_hf_corpus` |
| 4.3 | `test_train_on_unicode_text` | ✅ | Covered by `test_tiktoken_unicode_text`, `test_unicode_roundtrip` |
| 4.4 | `test_train_on_emojis` | 🟢 | Train on emoji text, verify encode/decode |
| 4.5 | `test_train_on_mixed_languages` | 🟢 | Mixed Greek/Cyrillic/CJK training |
| 4.6 | `test_train_on_empty_corpus_raises_valueerror` | 🟡 | Currently no error — trains 0 merges. Add validation? |
| 4.7 | `test_train_on_single_character_repeated` | 🟢 | Train on "aaaaaa..." |
| 4.8 | `test_train_on_single_word_repeated` | 🟢 | Train on "hellohellohello..." |
| 4.9 | `test_train_respects_vocab_size_limit` | ✅ | Covered by determinism tests |
| 4.10 | `test_train_vocab_size_smaller_than_initial_raises_valueerror` | 🟡 | Currently silent. Add validation? |
| 4.11 | `test_train_vocab_size_larger_than_achievable_terminates` | 🟢 | High vocab_size on tiny corpus → terminates cleanly |
| 4.12 | `test_train_is_deterministic` | ✅ | Covered |
| 4.13 | `test_train_tie_breaking_is_deterministic` | 🟢 | Equal-frequency pairs → consistent Tiebreak |
| 4.14 | `test_train_most_frequent_pair_merges_first` | 🟢 | Synthetic corpus, verify merge 0 is most frequent pair |
| 4.15 | `test_train_vocab_grows_one_token_per_merge` | 🟢 | Verify len(vocab) == 256 + len(merges) |
| 4.16 | `test_train_regex_pretokenization_splits_correctly` | ✅ | Split alignment tests |
| 4.17 | `test_train_merges_never_cross_pretoken_boundaries` | 🟢 | Verify no merge rule IDs cross split boundaries |
| 4.18 | `test_train_zero_or_negative_vocab_size_raises_valueerror` | 🟡 | Add validation |
| 4.19 | `test_train_none_corpus_raises_typeerror` | ❌ | No None in Mojo |
| 4.20 | `test_train_does_not_mutate_input_corpus` | 🟢 | Verify input unchanged after train |
| 4.21 | `test_retrain_reinitializes_state` | 🟢 | Train → train again → clean state (not appended) |

**New tests: 10 + 3 optional** (4.4, 4.5, 4.7, 4.8, 4.11, 4.13, 4.14, 4.15, 4.17, 4.20, 4.21 🟢; 4.6, 4.10, 4.18 🟡)

---

## Section 5 — INCREMENTAL PAIR STATISTICS

All 🔴 — internal implementation detail.

**Decision: Skip section 5.**

---

## Section 6 — ENCODING

| # | Test | Status | Notes |
|---|------|--------|-------|
| 6.1 | `test_encode_empty_string` | ✅ | Covered |
| 6.2 | `test_encode_single_byte` | 🟢 | Encode "a" → [97] |
| 6.3 | `test_encode_ascii_word` | ✅ | Covered |
| 6.4 | `test_encode_whitespace_variants` | 🟢 | Tab, newline, multiple spaces, NBSP |
| 6.5 | `test_encode_utf8_text` | ✅ | Covered |
| 6.6 | `test_encode_emoji_sequences` | 🟢 | |
| 6.7 | `test_encode_mixed_scripts_in_one_string` | 🟢 | Latin + CJK + Arabic + emoji |
| 6.8 | `test_encode_binary_bytes` | 🔴 | encode takes StringSlice — can hold arbitrary bytes |
| 6.9 | `test_encode_all_possible_byte_values` | 🟢 | String containing all 256 bytes |
| 6.10 | `test_encode_single_byte_not_in_vocab_falls_back_to_byte_token` | ✅ | Byte-level vocab always has all bytes |
| 6.11 | `test_encode_is_deterministic` | ✅ | Covered |
| 6.12 | `test_encode_matches_greedy_lowest_rank_merge` | 🟢 | Synthetic vocab + encode verification |
| 6.13 | `test_encode_long_repeated_substring_no_quadratic_blowup` | 🟧 | Hard to test in CI — skip for now |
| 6.14 | `test_encode_none_input_raises_typeerror` | ❌ | No None |
| 6.15 | `test_encode_non_string_input_raises_typeerror` | ❌ | Mojo is typed |

**New tests: 7** (6.2, 6.4, 6.6, 6.7, 6.9, 6.12, plus maybe 6.8)

---

## Section 7 — DECODING

| # | Test | Status | Notes |
|---|------|--------|-------|
| 7.1 | `test_decode_empty_tokens` | ✅ | Covered |
| 7.2 | `test_decode_single_token` | 🟢 | Decode [97] → "a" |
| 7.3 | `test_decode_multiple_tokens` | ✅ | Implied by roundtrip tests |
| 7.4 | `test_decode_utf8_reconstruction` | ✅ | Covered |
| 7.5 | `test_decode_emoji_reconstruction` | 🟢 | |
| 7.6 | `test_decode_binary_bytes` | 🔴 | decode returns String — can't represent arbitrary binary |
| 7.7 | `test_decode_whitespace_preservation` | 🟢 | |
| 7.8 | `test_decode_out_of_range_token_id_raises_indexerror` | 🟡 | Add bounds check in decode |
| 7.9 | `test_decode_mixed_valid_invalid_ids_raises_indexerror` | 🟡 | Same |
| 7.10 | `test_decode_partial_utf8_sequence_raises_or_uses_fallback` | 🔴 | Already uses from_utf8 (lossy) |
| 7.11–7.13 | None/non-int/non-list errors | ❌ | Mojo is typed |

**New tests: 4 + 2 optional** (7.2, 7.5, 7.7 🟢; 7.8, 7.9 🟡)

---

## Section 8 — ENCODE/DECODE ROUND-TRIP

| # | Test | Status | Notes |
|---|------|--------|-------|
| 8.1 | `test_encode_decode_roundtrip_ascii` | ✅ | Covered |
| 8.2 | `test_encode_decode_roundtrip_unicode` | ✅ | Covered |
| 8.3 | `test_encode_decode_roundtrip_emojis` | 🟢 | Combined with 6.6 |
| 8.4 | `test_encode_decode_roundtrip_mixed_language` | 🟢 | Combined with 6.7 |
| 8.5 | `test_encode_decode_roundtrip_large_document` | 🟢 | Use multi-KB corpus text |
| 8.6 | `test_encode_decode_roundtrip_random_bytes` | 🔴 | Not applicable |
| 8.7 | `test_encode_decode_roundtrip_with_special_tokens` | ✅ | Covered |
| 8.8 | `test_encode_decode_roundtrip_stable_across_repetitions` | 🟢 | encode→decode→encode→decode stability |

**New tests: 4** (8.3–8.5, 8.8 — 8.3/8.4 can share with section 6)

---

## Section 9 — SPECIAL TOKENS

| # | Test | Status | Notes |
|---|------|--------|-------|
| 9.1 | `test_special_tokens_get_reserved_ids` | ✅ | Covered |
| 9.2 | `test_special_tokens_not_split_by_bpe` | ✅ | Covered |
| 9.3 | `test_special_token_in_text_without_allowlist_raises_valueerror` | 🔴 | No disallowed_special concept |
| 9.4 | `test_special_token_explicit_allowlist_permits_encoding` | ✅ | Covered |
| 9.5 | `test_decode_special_token_id_returns_original_string` | ✅ | Covered |
| 9.6 | `test_special_token_empty_string_raises_valueerror` | 🟡 | Add validation for empty special token |
| 9.7 | `test_special_tokens_duplicate_raises_valueerror` | 🟡 | Currently overwrites silently. Add duplicate check? |
| 9.8 | `test_special_token_overlap_with_surrounding_text` | 🟢 | Text containing special-token-like substrings |

**New tests: 1 + 2 optional** (9.8 🟢; 9.6, 9.7 🟡)

---

## Section 10 — BYTE / UNICODE EDGE CASES

| # | Test | Status | Notes |
|---|------|--------|-------|
| 10.1 | `test_handles_embedded_null_byte` | 🟢 | String with \0 |
| 10.2 | `test_encode_malformed_utf8_does_not_raise` | 🟢 | Invalid UTF-8 bytes |
| 10.3 | `test_whitespace_only_input` | 🟢 | Space/tab/newline only |
| 10.4 | `test_punctuation_only_input` | 🟢 | Punctuation only |
| 10.5 | `test_very_long_unsplittable_token` | 🟢 | No spaces, very long |

**New tests: 5** (10.1–10.5 🟢)

---

## Section 11 — .TIKTOKEN FORMAT I/O

| # | Test | Status | Notes |
|---|------|--------|-------|
| 11.1 | `test_load_tiktoken_parses_all_lines` | ✅ | Covered |
| 11.2 | `test_load_tiktoken_ranks_match_line_order` | ✅ | Covered |
| 11.3 | `test_load_tiktoken_base64_decodes_correctly` | ✅ | Covered |
| 11.4 | `test_save_load_tiktoken_roundtrip` | ✅ | Covered |
| 11.5 | `test_load_tiktoken_malformed_line_raises_valueerror` | 🟡 | Malformed base64 / missing rank. Currently crashes. |
| 11.6 | `test_load_tiktoken_duplicate_token_raises_valueerror` | 🟡 | Currently overwrites. Add duplicate check. |
| 11.7 | `test_load_tiktoken_duplicate_merge_raises_valueerror` | 🟡 | Same. |
| 11.8 | `test_load_tiktoken_truncated_file_raises_valueerror` | 🟢 | Incomplete file |
| 11.9 | `test_load_tiktoken_unknown_version_raises_valueerror` | 🔴 | No version marker |
| 11.10 | `test_load_tiktoken_missing_file_raises_filenotfounderror` | 🟡 | Currently crashes on missing file |
| 11.11 | `test_load_tiktoken_path_traversal_raises_valueerror` | 🔴 | Would need path validation |
| 11.12 | `test_save_tiktoken_before_training_raises_valueerror` | 🟡 | Add "trained" flag or check |
| 11.13 | `test_encode_before_training_raises_valueerror` | 🟡 | Same |
| 11.14 | `test_decode_before_training_raises_valueerror` | 🟡 | Same |
| 11.15 | `test_tiktoken_format_matches_reference_gpt2_file` | 🟢 | Load real GPT-2 .tiktoken |
| 11.16 | `test_tiktoken_format_matches_reference_cl100k_file` | 🟢 | Load real cl100k .tiktoken |
| 11.17 | `test_load_tiktoken_large_vocab_within_time_budget` | ✅ | Covered by o200k test |

**New tests: 2 + 6 optional** (11.8, 11.15, 11.16 🟢; 11.5–11.7, 11.10, 11.12–11.14 🟡)

---

## Section 12 — CROSS-VALIDATION / COMPATIBILITY

| # | Test | Status | Notes |
|---|------|--------|-------|
| 12.1 | `test_matches_tiktoken_library_encoding_on_corpus` | 🟢 | Python cross-validation (complex) |
| 12.2 | `test_matches_tiktoken_library_decoding_on_corpus` | 🟢 | Same |
| 12.3 | `test_matches_reference_merge_order` | 🟢 | Compare trained merges (needs reference) |
| 12.4 | `test_matches_reference_token_count_on_large_document` | 🟢 | Same |
| 12.5 | `test_matches_reference_on_known_tricky_strings` | 🟢 | Curated string list |

**New tests: 5** (complex — defer to Phase 3)

---

## Section 13 — PROPERTY-BASED TESTS

| # | Test | Status | Notes |
|---|------|--------|-------|
| 13.1 | `test_property_encode_decode_identity` | ✅ | Covered (roundtrip tests) |
| 13.2 | `test_property_incremental_equals_naive_recount` | 🔴 | Internal detail |
| 13.3 | `test_property_training_is_deterministic` | ✅ | Covered |
| 13.4 | `test_property_vocab_ids_unique` | 🟢 | Same as 1.4 |
| 13.5 | `test_property_merge_graph_is_acyclic` | 🟡 | Walk merge graph for cycles |
| 13.6 | `test_property_merge_ranks_strictly_increasing` | 🟢 | Verify ranks follow training order |
| 13.7 | `test_property_every_token_is_decodable` | 🟢 | Every ID 0..len(vocab)-1 decodes cleanly |

**New tests: 3 + 1 optional** (13.4, 13.6, 13.7 🟢; 13.5 🟡)

---

## Section 14 — PERFORMANCE / REGRESSION GUARDS

All 🟧 — timing-based. Better suited as benchmarks.

| # | Test | Status | Notes |
|---|------|--------|-------|
| 14.1–14.5 | Throughput/time-budget tests | 🟧 | Already covered by `benchmarks/bm.mojo` |

**Decision: Skip section 14** (already have benchmarks).

---

## Section 15 — SECURITY / DOS

All 🟧 — timing-based, hard to test in CI.

| # | Test | Status | Notes |
|---|------|--------|-------|
| 15.1 | `test_adversarial_input_bounded_loop_time` | 🟧 | Defer |
| 15.2 | `test_extremely_long_text_raises_or_completes_within_budget` | 🟧 | Defer |

---

## Section 16 — REGRESSION TESTS

N/A yet — no historical bugs to pin.

---

## Summary Totals

| Priority | New Tests | Needs Source Change |
|----------|-----------|-------------------|
| 🟢 Phase 1 (safe — only new test code) | ~20 | 0 |
| 🟡 Phase 2 (adds validation/error handling) | ~7 | 7 source changes |
| 🟢 Phase 3 (edge cases, cross-validation) | ~8 | 0 (but complex) |
| 🔴 Defer/Skip | ~75 | — |

## Implementation Order

### Phase 1 — 20 tests, no source changes needed

1. `test_vocab_unique_ids` (1.4)
2. `test_vocab_size_matches_expected` (1.6)
3. `test_no_merge_possible_returns_input_unchanged` (3.4)
4. `test_train_on_emojis` (4.4)
5. `test_train_on_mixed_languages` (4.5)
6. `test_train_on_single_character_repeated` (4.7)
7. `test_train_on_single_word_repeated` (4.8)
8. `test_train_vocab_size_larger_than_achievable_terminates` (4.11)
9. `test_train_tie_breaking_is_deterministic` (4.13)
10. `test_train_most_frequent_pair_merges_first` (4.14)
11. `test_train_vocab_grows_one_token_per_merge` (4.15)
12. `test_train_merges_never_cross_pretoken_boundaries` (4.17)
13. `test_train_does_not_mutate_input_corpus` (4.20)
14. `test_retrain_reinitializes_state` (4.21)
15. `test_encode_single_byte` (6.2)
16. `test_encode_whitespace_variants` (6.4)
17. `test_encode_all_possible_byte_values` (6.9)
18. `test_decode_whitespace_preservation` (7.7)
19. `test_encode_decode_roundtrip_large_document` (8.5)
20. `test_encode_decode_roundtrip_stable_across_repetitions` (8.8)

### Phase 2 — 7+ tests, source validation changes needed

Source changes needed:
- `train()`: validate vocab_size ≥ 256, vocab_size > 0, empty corpus (optional)
- `_register_special_token()`: reject empty string, reject duplicates
- `decode()`: bounds check token IDs
- `load_tiktoken()`: catch malformed lines, duplicate tokens, missing file
- `save_tiktoken()`/`encode()`/`decode()`: detect untrained state

Tests:
1. `test_vocab_size_below_byte_range_raises_valueerror` (0.4)
2. `test_train_empty_corpus_raises_valueerror` (4.6)
3. `test_train_vocab_size_smaller_than_initial_raises_valueerror` (4.10)
4. `test_train_zero_or_negative_vocab_size_raises_valueerror` (4.18)
5. `test_decode_out_of_range_token_id_raises_indexerror` (7.8)
6. `test_decode_mixed_valid_invalid_ids_raises_indexerror` (7.9)
7. `test_special_token_empty_string_raises_valueerror` (9.6)

### Phase 3 — Complex / edge-case / cross-validation tests

1. `test_encode_emoji_sequences` (6.6) + `test_encode_decode_roundtrip_emojis` (8.3)
2. `test_encode_mixed_scripts_in_one_string` (6.7) + `test_encode_decode_roundtrip_mixed_language` (8.4)
3. `test_encode_matches_greedy_lowest_rank_merge` (6.12)
4. `test_decode_emoji_reconstruction` (7.5)
5. All section 10 byte/unicode edge cases (10.1–10.5)
6. `test_special_token_overlap_with_surrounding_text` (9.8)
7. `test_property_merge_ranks_strictly_increasing` (13.6)
8. `test_property_every_token_is_decodable` (13.7)
9. Cross-validation tests (section 12) — Python-dependent
