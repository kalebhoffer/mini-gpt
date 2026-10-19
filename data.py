"""
Step 1: get text, train the tokenizer on it, and save the text as token ids.

    python data.py                      # Tiny Shakespeare (~1MB, downloaded)
    python data.py --input my_text.txt  # or any plain-text file you like

Outputs, in data/:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to a .txt file (default: download Tiny Shakespeare)")
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    assert args.vocab_size <= 65536, "token ids are stored as uint16"

    os.makedirs(args.out_dir, exist_ok=True)
    path = args.input
    if path is None:
        path = os.path.join(args.out_dir, "input.txt")
        if not os.path.exists(path):
            print(f"downloading Tiny Shakespeare -> {path}")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)

    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"loaded {len(text):,} characters from {path}")

    print(f"training BPE tokenizer (vocab {args.vocab_size})...")
    t0 = time.time()
    tok = Tokenizer.train(text, args.vocab_size, verbose=True)
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    print(f"  done in {time.time() - t0:.1f}s")

    ids = np.array(tok.encode(text), dtype=np.uint16)
    split = int(0.9 * len(ids))
    ids[:split].tofile(os.path.join(args.out_dir, "train.bin"))
    ids[split:].tofile(os.path.join(args.out_dir, "val.bin"))

    print(f"{len(ids):,} tokens ({len(text) / len(ids):.2f} characters per token)")
    print(f"  train: {split:,}   val: {len(ids) - split:,}")
    sample = tok.encode("ROMEO: But soft, what light")
    print("example:", [tok.decode([i]) for i in sample])


if __name__ == "__main__":
    main()
