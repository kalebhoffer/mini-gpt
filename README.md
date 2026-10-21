# mini-gpt

A small GPT-style language model built from first principles in PyTorch. The tokenizer,
every network layer, the loss function, the optimizer, and the training loop are written
by hand. PyTorch supplies only tensors, autograd, and the Apple Silicon GPU backend.

The goal is understanding, not scale: every component is small enough to read in one
sitting, commented to explain *why* as well as *what*, and checked against PyTorch's
reference implementation.

> **Once upon a time, there was a dragon**. The dragon was very cute and loved to sing. One day, the dragon saw a little girl with a
> big smile. She wanted to sing her song.
> The dragon said to the girl, "I want to sing a song, please?" The girl smiled and said,
> "Of course! You can sing!"
> The girl sang her little song, and soon she heard a voice. The voice said, "Go away,
> dragon, you can sing too!" The dragon and the girl became friends. They sang songs
> together in the forest.

<sub>The prompt is in bold and the rest is generated. Output from an 11.5M-parameter model trained for 68 minutes on a MacBook (M4), using TinyStories data.</sub>

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
Run 3 sample:

```
ROMEO:
Even in Juliet,
Whose worse honour than his hands,
That I may promised and part
Of what's say, I have down a gentleman,
I' the receive a sweet cold course,
To some city to me.
```

- **Hand-written ops are slow.** ~1.4 s/step for 10.9M parameters. Every op launches its
  own GPU kernel, where PyTorch's fused attention would combine them. That's the cost of
  readability, and a fair trade for a learning project.

## Results: TinyStories (testing the data hypothesis)

The Shakespeare runs pointed to data as the bottleneck, so the next experiment kept the
architecture fixed and changed only the data. [TinyStories](https://arxiv.org/abs/2305.07759)
(Eldan & Li, 2023) is a corpus of short stories written by GPT-3.5/4 using vocabulary a young
child would know, designed to show that small models can learn coherent language.

`data.py` streams just the first 50MB of the 2.2GB file using an HTTP Range request:
64k stories, 14.1M tokens with a 2,048-token BPE vocabulary (3.65 chars/token), and
**24× more training data** than Tiny Shakespeare.

| Params | Width / heads / layers | Dropout | Steps | Tokens seen | Train loss | Val loss |
|---|---|---|---|---|---|---|
| 11.5M | 384 / 6 / 6 | 0.1 | 3,000 (68 min) | 24.6M (~1.9 epochs) | 1.726 | **1.774** |

- **No overfitting.** The train–val gap stayed under 0.05 for the whole run (versus 0.4–0.7
  at the best Shakespeare checkpoints), even with lighter dropout. Val loss was still falling at step 3,000.
- **Same model, much better output.** The Shakespeare model produces the right format with
  garbled sentences. The TinyStories model produces grammatical stories with named characters,
  dialogue, and a beginning, middle, and end. The architecture didn't change, only the data.
- Loss values aren't directly comparable across the two datasets (different tokenizers and
  vocab sizes). Judge by the samples.

```
Once upon a time, in a small house, there was a little girl named Lucy. Lucy loved to play
outside with her friends. One day, she found a big, red ball in her room. It was a fun ball
with a zip.
Lucy took the ball to the park to play with her friends. They played with the ball and had
a lot of fun. Her friends were happy too. Lucy and her friends played with the ball all day long.
```

Remaining failure modes are typical for this scale: occasional logic slips (a king appears
from nowhere at the end of the dragon story) and repetitive phrasing ("had a lot of fun").

## Run it

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch numpy pytest

python -m pytest tests -q

# TinyStories (recommended)
python data.py --dataset tinystories    # ~30s: download 50MB, train tokenizer, encode
python train.py --data-dir data/tinystories --out-dir checkpoints/tinystories \
    --max-iters 3000 --dropout 0.1      # ~68 min on M4
python sample.py --checkpoint checkpoints/tinystories/best.pt --prompt "Once upon a time"

# Tiny Shakespeare
python data.py                          # download, train tokenizer, encode
python train.py --quick                 # 20-second smoke test with a tiny model
python train.py --max-iters 2000 --dropout 0.3     # run 3 configuration (~47 min on M4)
python sample.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"
```

Train on any text: `python data.py --input path/to/file.txt --out-dir data/mine`, then
`python train.py --data-dir data/mine`.

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
his [char-rnn](https://github.com/karpathy/char-rnn). TinyStories is from Ronen Eldan and
Yuanzhi Li, [*TinyStories: How Small Can Language Models Be and Still Speak Coherent
English?*](https://arxiv.org/abs/2305.07759) (2023).
