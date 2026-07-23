"""BPE tokenizer — train, encode, decode, save, load."""

from std.pathlib import Path
from std.python import Python


struct PreTokenizer:
    @staticmethod
    def tokenize[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        var splits = (
            StringSlice(text)
            .replace(" ", " " + spacer)
            .replace(".", " .")
            .split(" ")
        )
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    var vocab: List[String]
    var stoi: Dict[String, Int]
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        # 1. Pre-tokenize and compute word frequencies
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.tokenize(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # 2. Build the alphabet from all unique characters
        var alphabet: List[String] = []
        for word in word_freqs.keys():
            for letter in word.codepoints():
                var char_str = chr(Int(letter))
                if char_str not in alphabet:
                    alphabet.append(char_str)
        sort(alphabet)

        # 3. Initialize vocab: special token + every character
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # 4. Split each word into individual characters
        var splits = Dict[String, List[String]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(c)) for c in word.codepoints()]

        # 5. Iteratively merge the most frequent pair
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            if len(pair_freqs) == 0:
                break
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        if text.byte_length() == 0:
            return List[String]()
        var words = PreTokenizer.tokenize(text)
        var splits = [
            [chr(Int(code)) for code in word.codepoints()]
            for word in words
        ]
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        if len(ids) == 0:
            return String("")
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        return len(self.vocab)

    def save(self, path: String) raises:
        var json = Python.import_module("json")
        var data = Python.dict()

        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        splits[word] = split.copy()
