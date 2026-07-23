from tokenizer import BPETokenizer


def check(condition: Bool, label: String) raises -> Bool:
    if condition:
        print("  PASS:", label)
        return True
    else:
        print("  FAIL:", label)
        return False


def test_unk_fallback() raises -> Bool:
    """Train on a small corpus, then encode chars not in training set."""
    var corpus = List[String]()
    corpus.append(String("abc"))
    var tok = BPETokenizer()
    tok.train(corpus, 12)
    # "xyz" are not in the alphabet {'a','b','c',Ġ} — each maps to UNK (ID 0)
    var ids = tok.encode(String("xyz"))
    var ok = True
    ok = check(
        len(ids) == 3 and ids[0] == 0 and ids[1] == 0 and ids[2] == 0,
        "unknown chars mapped to UNK (0)",
    ) and ok
    ok = check(
        tok.decode(ids) == "<UNK><UNK><UNK>",
        "UNK IDs decode to visible <UNK> text",
    ) and ok
    return ok


def test_basic_roundtrip() raises -> Bool:
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 30)

    var ok = True
    ok = check(len(tok) == 18, "small corpus: vocab stops at 18 (exhausted merges)") and ok
    ok = check(
        tok.decode(List[Int](length=1, fill=0)) == "<UNK>",
        "ID 0 decodes to <UNK>",
    ) and ok

    var ids = tok.encode(String("hello world"))
    ok = check(len(ids) > 0, "encode returns tokens") and ok
    ok = check(
        tok.decode(ids) == "hello world",
        "roundtrip preserves input",
    ) and ok
    return ok


def test_empty_input() raises -> Bool:
    var corpus = List[String]()
    corpus.append(String("abc"))
    var tok = BPETokenizer()
    tok.train(corpus, 12)

    var ok = True
    var empty_ids = tok.encode(String(""))
    ok = check(len(empty_ids) == 0, "empty string → empty ids") and ok
    var empty_decoded = tok.decode(List[Int]())
    ok = check(empty_decoded.byte_length() == 0, "empty ids → empty string") and ok
    return ok


def test_save_load() raises -> Bool:
    var corpus = List[String]()
    corpus.append(String("hello world"))
    var tok = BPETokenizer()
    tok.train(corpus, 30)

    tok.save("/tmp/bpe_test.json")
    var loaded = BPETokenizer.load("/tmp/bpe_test.json")

    var ok = True
    ok = check(len(loaded) == len(tok), "loaded vocab size matches") and ok

    var text = String("hello world")
    var loaded_ids = loaded.encode(text)
    ok = check(
        loaded.decode(loaded_ids) == "hello world",
        "loaded tokenizer roundtrip OK",
    ) and ok
    return ok


def test_deterministic() raises -> Bool:
    var corpus = List[String]()
    corpus.append(String("hello world"))

    var tok1 = BPETokenizer()
    tok1.train(corpus, 30)
    var tok2 = BPETokenizer()
    tok2.train(corpus, 30)

    return check(
        len(tok1.merges) == len(tok2.merges),
        "same corpus → same merge count",
    )


def test_full_hf_corpus() raises -> Bool:
    var corpus = List[String]()
    corpus.append(String("This is the Hugging Face Course."))
    corpus.append(String("This chapter is about tokenization."))
    corpus.append(String("This section shows several tokenizer algorithms."))
    corpus.append(String(
        "Hopefully, you will be able to understand how they are trained and"
        " generate tokens."
    ))
    var tok = BPETokenizer()
    tok.train(corpus, 50)

    var ok = True
    ok = check(len(tok) == 50, "trains to vocab_size=50") and ok

    var text = String("This is not a token.")
    var ids = tok.encode(text)
    ok = check(
        tok.decode(ids) == "This is not a token.",
        "HF corpus roundtrip OK",
    ) and ok

    # Save/load the bigger model too
    tok.save("/tmp/bpe_hf_test.json")
    var loaded = BPETokenizer.load("/tmp/bpe_hf_test.json")
    ok = check(len(loaded) == 50, "loaded HF vocab size OK") and ok

    var loaded_ids = loaded.encode(text)
    ok = check(
        loaded.decode(loaded_ids) == "This is not a token.",
        "loaded HF roundtrip OK",
    ) and ok
    return ok


def main() raises:
    var passed = 0
    var total = 0

    if test_unk_fallback():
        passed += 1
    total += 1

    if test_basic_roundtrip():
        passed += 1
    total += 1

    if test_empty_input():
        passed += 1
    total += 1

    if test_save_load():
        passed += 1
    total += 1

    if test_deterministic():
        passed += 1
    total += 1

    if test_full_hf_corpus():
        passed += 1
    total += 1

    print()
    print("========================")
    print("  ", passed, "/", total, "suites passed")
    if passed < total:
        print("  ", total - passed, "FAILURES")
    print("========================")
