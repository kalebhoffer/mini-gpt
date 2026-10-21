# mini-gpt

A small GPT-style language model built from first principles in PyTorch. The tokenizer,
every network layer, the loss function, the optimizer, and the training loop are written
by hand. PyTorch supplies only tensors, autograd, and the Apple Silicon GPU backend.

The goal is understanding, not scale: every component is small enough to read in one
sitting, commented to explain *why* as well as *what*, and checked against PyTorch's
reference implementation.

```
ROMEO:
Even in Juliet,
Whose worse honour than his hands,
That I may promised and part
Of what's say, I have down a gentleman,
I' the receive a sweet cold course,
To some city to me.

CORIOLANUS:
Well, must I know how I do,
You shall be safe of the dearest,
```
<sub>Output from a 10.9M-parameter model after 47 minutes of training on a MacBook (M4).</sub>

## What's implemented

| Component | File | Notes |
|---|---|---|
| Byte-level BPE tokenizer | `tokenizer.py` | GPT-2-style pre-tokenization; trains on unique chunks weighted by frequency instead of the raw text (~4s on Tiny Shakespeare) |
| Linear, Embedding, LayerNorm | `model.py` | Plain `nn.Parameter` tensors and matrix math |
| Softmax, GELU, dropout, cross-entropy | `model.py` | Numerically stable (max-subtraction / log-sum-exp) |
| Multi-head causal self-attention | `model.py` | Explicit Q/K/V projection, scaled dot product, causal mask |
| Transformer block + GPT | `model.py` | Pre-norm residual blocks, learned positional embeddings, tied input/output embeddings, GPT-2 scaled init |
| AdamW + gradient clipping | `optim.py` | Bias-corrected moments, decoupled weight decay (matrices only) |
| Training loop | `train.py` | Linear warmup + cosine decay, held-out evaluation, best-checkpoint selection |
| Sampling | `sample.py` | Temperature and top-k |

### Verified against PyTorch

`tests/` checks each hand-written piece against PyTorch's built-in version (13 tests, ~3s):

- softmax, GELU, cross-entropy, and LayerNorm match `torch` / `torch.nn.functional` numerically
- AdamW matches `torch.optim.AdamW` parameter-for-parameter over 10 steps with weight decay
- gradient clipping matches `torch.nn.utils.clip_grad_norm_`
- **causality:** changing a future token never changes predictions for earlier positions
- initial loss equals `ln(vocab_size)` (a correctly initialized model starts at uniform guessing)
- the full model plus optimizer can memorize a fixed batch (an end-to-end check that gradients flow)
- the tokenizer round-trips arbitrary text losslessly, including Unicode it never saw in training

## Results: Tiny Shakespeare

Data: 1.1M characters → 575k tokens (BPE vocab 512, 1.94 chars/token), split 90/10.
Hardware: Apple M4, MPS backend, fp32, batch 32 × 256 tokens.

| Run | Params | Width / heads | Dropout | Steps | Best val loss | Diagnosis |
|---|---|---|---|---|---|---|
| 1 | 10.9M | 384 / 6 | 0.2 | 2,600 of 5,000 (stopped) | 2.858 @ 1,250 | Overfit: val loss bottomed at step 1,250 and rose while train loss kept falling (1.44 vs 3.04 by step 2,500) |
| 2 | 4.9M | 256 / 8 | 0.3 | 1,500 | 2.999 @ 1,500 | Underfit: too little capacity and too few steps; val loss was still falling when training ended |
| **3** | **10.9M** | **384 / 6** | **0.3** | **2,000** | **2.830 @ 2,000** | **Best: full capacity, and heavier regularization held off overfitting** |

Takeaways:

- **Overfitting and underfitting are both visible at this scale.** Run 1 memorized the
  training text, and run 2 over-corrected. Run 3 combined run 1's capacity with run 2's
  regularization and beat both.
- **Data, not tuning, is the bottleneck.** Three runs of hyperparameter changes moved val
  loss by about 1%. With 518k training tokens, a 10.9M-parameter model sees each token
  about 16 times every 1,000 steps (32 × 256 × 1,000 / 518k) and memorizes quickly.
  Scaling the data is the next lever.
- **Hand-written ops are slow.** ~1.4 s/step for 10.9M parameters. Every op launches its
  own GPU kernel, where PyTorch's fused attention would combine them. That's the cost of
  readability, and a fair trade for a learning project.

## Run it

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch numpy pytest

python data.py                          # download Tiny Shakespeare, train tokenizer, encode
python train.py --quick                 # 20-second smoke test with a tiny model
python train.py --max-iters 2000 --dropout 0.3     # run 3 configuration (~47 min on M4)
python sample.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"
python -m pytest tests -q
```

Train on any text: `python data.py --input path/to/file.txt`.

## Reading order

| # | File | What it covers |
|---|------|----------------|
| 1 | `tokenizer.py` | How text becomes numbers: byte-pair encoding |
| 2 | `model.py` (top) | Primitives: softmax, GELU, dropout, cross-entropy |
| 3 | `model.py` (layers) | Linear, Embedding, LayerNorm |
| 4 | `model.py` `CausalSelfAttention` | The core mechanism: tokens attending to earlier tokens |
| 5 | `model.py` `Block`, `GPT` | Residual stacking, weight tying, generation |
| 6 | `optim.py` | AdamW and gradient clipping |
| 7 | `train.py` | Training loop, LR schedule, overfitting detection |
| 8 | `tests/` | How each piece is verified |

## Experiments to try

- **Remove the causal mask** (`masked_fill` in attention). Train loss plummets because the
  model reads the answer from the future, and generation collapses. Why?
- **Remove residual connections** (`x = x + self.attn(...)` → `x = self.attn(...)`).
- **Width vs depth:** `--n-layer 12 --n-embd 256` vs `--n-layer 4 --n-embd 512`.
- **Tokenizer size:** `--vocab-size 256` (raw bytes) vs `2048`.

## Out of scope (deliberately)

KV caching, fused attention kernels, mixed precision, distributed training, and
instruction tuning / RLHF. Each is a natural next chapter.

## Acknowledgments

The architecture follows GPT-2 (Radford et al., 2019). Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and *Zero to Hero* series are the standard
references for this kind of from-scratch build, and the Tiny Shakespeare dataset comes from
his [char-rnn](https://github.com/karpathy/char-rnn).
