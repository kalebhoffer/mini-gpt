"""
Byte-level BPE (Byte Pair Encoding) tokenizer, written from scratch.

A neural net can't read text, only numbers. A tokenizer turns text into a list
of integer ids ("tokens") and back again.

How BPE works:
  1. Start with the 256 possible byte values as the base vocabulary. Any text,
     in any language, is just a sequence of UTF-8 bytes, so nothing is ever
     "unknown".
  2. Count every adjacent pair of tokens in the training text.
  3. Take the most frequent pair (say "t" + "h") and give it a brand new id.
     Replace every occurrence of that pair with the new id.
  4. Repeat until the vocabulary is the size you want.

Common chunks like "the", "ing", " ROMEO" end up as single tokens, so the model
sees shorter sequences and each token carries more meaning. This is the same
idea GPT-2/3/4 use, just with a smaller vocabulary.
"""

import json
import re
from collections import Counter

# Before running BPE we split text into chunks (roughly words, numbers,
# punctuation runs, whitespace) so merges never cross word boundaries. Without
# this, "dog." "dog!" and "dog?" would each waste a vocab slot. This is a
# simplified, ASCII version of the regex GPT-2 uses. Every character matches
# one of the alternatives, so joining the chunks back together gives exactly
# the original text.
SPLIT_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"   # contractions: 's 'd 'm 't 'll 've 're
    r"| ?[A-Za-z]+"           # a word, with its leading space
    r"| ?[0-9]+"              # a number, with its leading space
    r"| ?[^\sA-Za-z0-9]+"     # punctuation / other symbols
    r"|\s+(?!\S)"             # trailing whitespace
    r"|\s+"                   # any other whitespace
)


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: dict[tuple[int, int], int] | None = None):
        # merges maps (left_id, right_id) -> new_id, in the order they were
        # learned. Python dicts keep insertion order, which we rely on.
        self.merges = merges or {}
        self._cache: dict[str, list[int]] = {}
        # vocab maps id -> the raw bytes it stands for (used by decode).
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "Tokenizer":
        assert vocab_size >= 256, "vocab must at least hold the 256 raw bytes"

        # Speed trick: Shakespeare has ~1M characters but only ~25k distinct
        # chunks. Run BPE once per distinct chunk, weighting each pair count by
        # how often the chunk appears, instead of walking the whole text.
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        merges = {}
        for new_id in range(256, vocab_size):
            pair_counts: Counter = Counter()
            for ids, freq in zip(words, freqs):
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break  # every chunk is already a single token
            best = max(pair_counts, key=pair_counts.get)
            merges[best] = new_id
            words = [merge(ids, best, new_id) for ids in words]
            if verbose and (new_id - 255) % 64 == 0:
                tok = cls(merges)
                print(f"  merge {new_id - 255:4d}/{vocab_size - 256}: "
                      f"{tok.vocab[new_id]!r} (seen {pair_counts[best]:,}x)")
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply merges in the same order training learned them: always the
            # pair with the lowest new_id first.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing left to merge
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for chunk in SPLIT_PATTERN.findall(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            out.extend(self._cache[chunk])
        return out

    def decode(self, ids: list[int]) -> str:
        # A sampled sequence can cut a multi-byte character in half, so replace
        # invalid UTF-8 instead of crashing.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path) as f:
            data = json.load(f)
        return cls({(a, b): idx for a, b, idx in data["merges"]})
