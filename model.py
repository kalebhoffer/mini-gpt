"""
A GPT-style transformer language model, written from scratch.

The model's one job: given a sequence of tokens, predict the next token. That
single skill, repeated, is how it writes text.

From PyTorch we only use tensors, autograd (automatic gradients), and
nn.Module/nn.Parameter (a container that tracks which tensors are learnable).
Every layer below (Linear, Embedding, LayerNorm, softmax, GELU, dropout,
attention, cross-entropy) is implemented by hand.

Data flow for a batch of B sequences, each T tokens long, with C = n_embd:

    token ids (B, T)
      -> token embedding + position embedding      (B, T, C)
      -> N x Block:  x = x + Attention(LayerNorm(x))
                     x = x + MLP(LayerNorm(x))
      -> final LayerNorm                           (B, T, C)
      -> project to vocab ("logits")               (B, T, vocab_size)

logits[b, t] is the model's score for every possible next token after position t.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 512   # number of distinct tokens (from the tokenizer)
    block_size: int = 256   # context length: max tokens the model can look back over
    n_layer: int = 6        # number of transformer blocks stacked
    n_head: int = 6         # attention heads per block
    n_embd: int = 384       # width of every token's vector ("embedding dimension")
    dropout: float = 0.2    # fraction of activations randomly zeroed during training


# ---------------------------------------------------------------------------
# Primitive functions
# ---------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Turn arbitrary scores into probabilities that are positive and sum to 1."""
    # exp() of a large number overflows. Subtracting the max first doesn't
    # change the result (it cancels in the division) but keeps numbers small.
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Smooth non-linearity used by GPT-2: roughly x for x>0, roughly 0 for x<0.
    Without a non-linearity, stacking layers would collapse into one big linear
    function and the model couldn't learn anything interesting."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Randomly zero a fraction p of values during training, so the model can't
    lean too hard on any single feature (reduces overfitting). Survivors are
    scaled up by 1/(1-p) so the expected value is unchanged. Off at eval time."""
    if not training or p == 0.0:
        return x
    keep = (torch.rand_like(x) >= p).to(x.dtype)
    return x * keep / (1.0 - p)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The training loss: -log(probability the model gave to the correct token),
    averaged over every position.

    logits: (N, vocab_size) raw scores; targets: (N,) correct token ids.

    log(softmax(x)[target]) = x[target] - log(sum(exp(x))), and we compute the
    log-sum-exp with the same max-subtraction trick as softmax.
    """
    m = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = m.squeeze(-1) + (logits - m).exp().sum(dim=-1).log()
    correct = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (log_sum_exp - correct).mean()


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Linear(nn.Module):
    """y = x @ W^T + b. The basic learnable building block: every output is a
    weighted sum of every input."""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, std: float = 0.02):
        super().__init__()
        # Small random init: big enough to break symmetry between neurons,
        # small enough that signals don't blow up through many layers.
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * std)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


class Embedding(nn.Module):
    """A lookup table: row i is the learned vector for token (or position) i."""

    def __init__(self, n_items: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_items, dim) * 0.02)

    def forward(self, idx):
        return self.weight[idx]


class LayerNorm(nn.Module):
    """Rescale each token's vector to mean 0, variance 1, then apply a learned
    scale and shift. Keeps activations in a stable range so deep stacks train."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class CausalSelfAttention(nn.Module):
    """The core idea of the transformer: every token looks back at the earlier
    tokens and pulls in information from the ones that are relevant to it.

    Each token emits three vectors:
      query (q): "what am I looking for?"
      key   (k): "what do I contain?"
      value (v): "what will I hand over if you attend to me?"
    The score between token i and token j is q_i . k_j. Softmax turns the
    scores into weights, and token i's output is the weighted sum of the v's.

    "Causal" means a token may only look at itself and earlier tokens, never
    the future. Otherwise predicting the next token would be cheating.

    "Multi-head": we split C into n_head smaller heads that each run attention
    independently, so different heads can track different relationships
    (one follows who is speaking, another rhyme, another grammar...).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly into heads"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        # One matrix computes q, k and v together (it's faster than three).
        self.qkv = Linear(cfg.n_embd, 3 * cfg.n_embd)
        # Layers that write into the residual stream get a smaller init so the
        # sum over many blocks starts out small (GPT-2 trick).
        self.proj = Linear(cfg.n_embd, cfg.n_embd, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # Lower-triangular True/False grid: mask[i, j] is True if j <= i.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        # (B, T, C) -> (B, n_head, T, head_dim) so each head works independently
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Every query against every key: (B, nh, T, hd) @ (B, nh, hd, T) -> (B, nh, T, T).
        # Dividing by sqrt(head_dim) keeps the scores from growing with head
        # size, which would make softmax too spiky to learn from.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Block the future: -inf becomes probability 0 after softmax.
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = softmax(att, dim=-1)
        att = dropout(att, self.dropout, self.training)

        y = att @ v                                       # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # glue the heads back together
        return dropout(self.proj(y), self.dropout, self.training)


class MLP(nn.Module):
    """Per-token "thinking" step. Attention moves information between tokens;
    the MLP then processes each token on its own. Expand to 4x width, apply a
    non-linearity, project back down."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = Linear(4 * cfg.n_embd, cfg.n_embd, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.dropout = cfg.dropout

    def forward(self, x):
        return dropout(self.proj(gelu(self.fc(x))), self.dropout, self.training)


class Block(nn.Module):
    """One transformer block. The `x = x + ...` residual connections matter:
    each block adds a small edit to the token vectors instead of replacing
    them, and gradients get a clean path straight back to the first layers."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = Embedding(cfg.vocab_size, cfg.n_embd)
        # Attention by itself has no idea of word order, so every position
        # 0..block_size-1 also gets a learned vector added in.
        self.pos_emb = Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)
        # No separate output matrix: we reuse the token embedding table
        # ("weight tying"). A token's input vector doubles as its output
        # "template", which saves parameters and usually trains better.

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)        # (B, T, C)
        x = dropout(x, self.cfg.dropout, self.training)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten so each (sequence, position) pair is one prediction.
            loss = cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Write text one token at a time: predict, sample, append, repeat.
        Call model.eval() first so dropout is off.

        temperature < 1 makes the model more conservative (sharper probabilities),
        > 1 more random. top_k only lets it pick among its k best guesses.
        """
        assert temperature > 0, "temperature must be > 0"
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]      # only the last block_size tokens fit
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # we only need the final position
            if top_k is not None:
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
