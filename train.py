"""
Step 2: train the model.

    python train.py            # full run (~10M parameters)
    python train.py --quick    # tiny model, ~1 minute, to check everything works

The loop, over and over:
  1. grab a random batch of text chunks, x, and the same chunks shifted one
     token to the right, y (y[t] is the "next token" after x[t])
  2. forward pass: the model predicts, cross-entropy scores how wrong it was
  3. backward pass: autograd computes how each weight affected the loss
  4. optimizer step: nudge every weight to make the loss a bit smaller

Every so often we measure loss on the held-out val set. When train loss keeps
falling but val loss starts rising, the model is memorizing instead of learning
(overfitting), so we keep the checkpoint with the best val loss.

Reading the loss: it's -log(probability of the right token). At the start it's
about log(vocab_size) = 6.2, which means random guessing over 512 tokens.
exp(loss) is roughly "how many tokens the model is torn between" at each step.
"""

import argparse
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig
from optim import AdamW, clip_grad_norm
from tokenizer import Tokenizer


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits: dict, args, device) -> dict:
    """Average loss over several batches. One batch alone is too noisy."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, args.batch_size, args.block_size, device))[1].item()
                  for _ in range(args.eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def get_lr(it: int, args) -> float:
    """Learning-rate schedule: ramp up linearly for `warmup` steps (big early
    steps on random weights are unstable), then glide down along a cosine curve
    to min_lr (small steps at the end to settle into a good spot)."""
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    # model shape
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.2)
    # training
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=get_device())
    args = ap.parse_args()

    if args.quick:
        args.block_size, args.n_layer, args.n_head, args.n_embd = 64, 2, 2, 64
        args.batch_size, args.max_iters, args.eval_interval, args.eval_iters = 16, 300, 100, 10
        args.out_dir = os.path.join(args.out_dir, "quick")

    torch.manual_seed(args.seed)
    device = args.device

    tok = Tokenizer.load(os.path.join(args.data_dir, "tokenizer.json"))
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters on {device}")
    print(f"data:  {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # Weight decay on the big matrices only. Shrinking biases and LayerNorm
    # gains toward zero doesn't help and can hurt.
    params = list(model.parameters())
    opt = AdamW([
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ])

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    newline = torch.tensor([tok.encode("\n")], device=device)
    t0 = time.time()

    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, args, device)
            marker = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg),
                            "iter": it, "val_loss": best_val},
                           os.path.join(args.out_dir, "best.pt"))
                marker = "  (saved best.pt)"
            print(f"\n== step {it}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}{marker}")
            model.eval()
            sample = model.generate(newline, 120, temperature=0.8, top_k=50)[0].tolist()
            model.train()
            print("-- sample --\n" + tok.decode(sample).strip() + "\n------------")
        if it == args.max_iters:
            break

        lr = get_lr(it, args)
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(params, args.grad_clip)
        opt.step(lr)

        if it % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {it:5d} | loss {loss.item():.3f} | lr {lr:.2e} | grad norm {grad_norm:.2f} "
                  f"| {elapsed / max(it, 1) * 1000:.0f} ms/step")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min, best val loss {best_val:.3f}")
    print(f"try it:  python sample.py --checkpoint {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
