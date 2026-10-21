"""
Step 1: get text, train the tokenizer on it, and save the text as token ids.

    python data.py                                   # Tiny Shakespeare (~1MB)
    python data.py --dataset tinystories             # first 50MB of TinyStories
    python data.py --input my_text.txt --out-dir data/mine   # any plain-text file

Outputs, in data/<dataset>/:
    tokenizer.json   the learned BPE merges
    train.bin        first 90% of the tokens (uint16), what the model learns from
    val.bin          last 10%, held out so we can check it isn't just memorizing
"""

import argparse
import os
import time
import urllib.request

import numpy as np

from tokenizer import Tokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# TinyStories (Eldan & Li, 2023): short stories written by GPT-3.5/4 using only words a
# 3-4 year old would know. Small models learn grammar and plot from it far better than
# from Shakespeare, because the language is simple and there's a lot of it.
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
STORY_SEPARATOR = "<|endoftext|>"

DEFAULT_VOCAB = {"shakespeare": 512, "tinystories": 2048}


def download_shakespeare(path: str) -> None:
    print(f"downloading Tiny Shakespeare -> {path}")
    urllib.request.urlretrieve(SHAKESPEARE_URL, path)


def download_tinystories(path: str, max_mb: int) -> None:
    # The full file is 2.2GB, far more than a ~10M parameter model needs. An
    # HTTP Range request fetches just the first max_mb megabytes.
    print(f"downloading first {max_mb}MB of TinyStories -> {path}")
    req = urllib.request.Request(TINYSTORIES_URL, headers={"Range": f"bytes=0-{max_mb * 2**20 - 1}"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    # The cut can land mid-character or mid-story: drop the partial story at the end.
    text = raw.decode("utf-8", errors="ignore")
    text = text[:text.rfind(STORY_SEPARATOR)]
    stories = [s.strip() for s in text.split(STORY_SEPARATOR) if s.strip()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stories) + "\n")
    print(f"  {len(stories):,} stories")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["shakespeare", "tinystories"], default="shakespeare")
    ap.add_argument("--input", help="path to your own .txt file (overrides --dataset)")
    ap.add_argument("--max-mb", type=int, default=50, help="TinyStories: megabytes to download")
    ap.add_argument("--vocab-size", type=int, help="default: 512 for Shakespeare, 2048 for TinyStories")
    ap.add_argument("--out-dir", help="default: data/<dataset>")
    args = ap.parse_args()

    vocab_size = args.vocab_size or DEFAULT_VOCAB[args.dataset]
    assert vocab_size <= 65536, "token ids are stored as uint16"
    out_dir = args.out_dir or os.path.join("data", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    path = args.input
    if path is None:
        path = os.path.join(out_dir, "input.txt")
        if not os.path.exists(path):
            if args.dataset == "shakespeare":
                download_shakespeare(path)
            else:
                download_tinystories(path, args.max_mb)

    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"loaded {len(text):,} characters from {path}")

    print(f"training BPE tokenizer (vocab {vocab_size})...")
    t0 = time.time()
    tok = Tokenizer.train(text, vocab_size, verbose=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"  done in {time.time() - t0:.1f}s")

    print("encoding...")
    ids = np.array(tok.encode(text), dtype=np.uint16)
    split = int(0.9 * len(ids))
    ids[:split].tofile(os.path.join(out_dir, "train.bin"))
    ids[split:].tofile(os.path.join(out_dir, "val.bin"))

    print(f"{len(ids):,} tokens ({len(text) / len(ids):.2f} characters per token)")
    print(f"  train: {split:,}   val: {len(ids) - split:,}")
    sample = tok.encode(text[:60])
    print("example:", [tok.decode([i]) for i in sample])


if __name__ == "__main__":
    main()
