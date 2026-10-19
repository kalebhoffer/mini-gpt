# mini-gpt

A small GPT-style language model built from scratch to learn how LLMs work.
The tokenizer, every layer, the loss function, the optimizer, and the training
loop are all hand-written. PyTorch only supplies tensors, autograd, and the
Apple Silicon GPU.

## Run it

```bash
source .venv/bin/activate
python data.py                 # download Tiny Shakespeare, train tokenizer, encode
python train.py --quick        # 20-second smoke test
python train.py                # real run, ~10M parameters
python sample.py --prompt "ROMEO:" --tokens 300
python -m pytest tests -q      # check each hand-written piece against PyTorch
```

Train on your own text: `python data.py --input path/to/file.txt`, then `python train.py`.

## Reading order

| # | File | What you'll learn |
|---|------|-------------------|
| 1 | `tokenizer.py` | How text becomes numbers: byte-pair encoding |
| 2 | `model.py` (top) | Primitives: softmax, GELU, dropout, cross-entropy loss |
| 3 | `model.py` (layers) | Linear, Embedding, LayerNorm: the building blocks |
| 4 | `model.py` `CausalSelfAttention` | **The key idea**: tokens looking back at each other |
| 5 | `model.py` `Block`, `GPT` | Stacking blocks with residual connections; generation |
| 6 | `optim.py` | How weights actually change: AdamW, gradient clipping |
| 7 | `train.py` | The training loop, learning-rate schedule, overfitting |
| 8 | `tests/` | Proof that each piece matches PyTorch's built-in version |

## Experiments to try

- **Width vs depth:** `--n-layer 12 --n-embd 256` against `--n-layer 4 --n-embd 512`.
- **Context length:** `--block-size 64`. Does it lose track of who is speaking?
- **Break attention:** comment out the `masked_fill` line in attention. Train loss
  plummets (it's cheating by reading the future) and generation turns to garbage. Why?
- **Remove residuals:** change `x = x + self.attn(...)` to `x = self.attn(...)`.
- **Tokenizer size:** `python data.py --vocab-size 256` (characters only) against `2048`.
- **Overfitting:** `--dropout 0.0` and watch the gap between train and val loss widen.
- **Bigger data:** TinyStories (children's stories written by GPT-4) produces much more
  coherent output at this size than Shakespeare does.

## What's left out (on purpose)

Real GPTs add a KV cache (faster generation), fused attention kernels, mixed precision,
multi-GPU training, and instruction tuning / RLHF (what turns a text predictor into a
chat assistant). Each would be a good next chapter.
