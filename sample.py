"""
Step 3: have the trained model write text.

    python sample.py --prompt "ROMEO:" --tokens 300
    python sample.py --temperature 1.2      # wilder
    python sample.py --temperature 0.5      # safer, more repetitive
    python sample.py --checkpoint checkpoints/tinystories/best.pt --prompt "Once upon a time"
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import Tokenizer
from train import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", help="default: the tokenizer of the data the checkpoint trained on")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default=get_device())
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = GPT(GPTConfig(**ckpt["config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (step {ckpt['iter']}, val loss {ckpt['val_loss']:.3f})\n")

    # Checkpoints from before data_dir was saved all trained on Shakespeare.
    tokenizer_path = args.tokenizer or os.path.join(ckpt.get("data_dir", "data/shakespeare"), "tokenizer.json")
    tok = Tokenizer.load(tokenizer_path)
    prompt = torch.tensor([tok.encode(args.prompt or "\n")], device=args.device)
    for i in range(args.num_samples):
        out = model.generate(prompt, args.tokens, temperature=args.temperature, top_k=args.top_k)
        print(tok.decode(out[0].tolist()))
        if i < args.num_samples - 1:
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
