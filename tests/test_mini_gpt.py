"""Checks that each hand-written piece is correct, mostly by comparing it
against PyTorch's built-in version of the same thing."""

import math

import pytest
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, LayerNorm, cross_entropy, gelu, softmax
from optim import AdamW, clip_grad_norm
from tokenizer import Tokenizer

TEXT = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
        "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n") * 20


@pytest.fixture
def tiny_cfg():
    return GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


# --- tokenizer ---------------------------------------------------------------

def test_tokenizer_roundtrip_and_compression():
    tok = Tokenizer.train(TEXT, vocab_size=300)
    assert tok.vocab_size == 300
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8")) / 2  # merges actually compress


def test_tokenizer_handles_unseen_text_and_unicode():
    tok = Tokenizer.train(TEXT, vocab_size=300)
    weird = "Zebra's café — 42  tabs\tand  spaces 😀\n"
    assert tok.decode(tok.encode(weird)) == weird


def test_tokenizer_save_load(tmp_path):
    tok = Tokenizer.train(TEXT, vocab_size=280)
    tok.save(tmp_path / "tok.json")
    loaded = Tokenizer.load(tmp_path / "tok.json")
    assert loaded.merges == tok.merges
    assert loaded.encode(TEXT) == tok.encode(TEXT)


# --- primitives match PyTorch's versions -------------------------------------

def test_softmax_matches_torch():
    x = torch.randn(4, 10) * 50  # large values exercise the stability trick
    assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-6)


def test_gelu_matches_torch():
    x = torch.randn(1000) * 3
    assert torch.allclose(gelu(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_cross_entropy_matches_torch():
    logits, targets = torch.randn(64, 50) * 10, torch.randint(50, (64,))
    assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-5)


def test_layernorm_matches_torch():
    ln = LayerNorm(32)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-1, 1)
    x = torch.randn(4, 7, 32)
    expected = F.layer_norm(x, (32,), ln.weight, ln.bias, eps=1e-5)
    assert torch.allclose(ln(x), expected, atol=1e-5)


# --- model -------------------------------------------------------------------

def test_model_is_causal(tiny_cfg):
    """Changing a future token must not change predictions for earlier positions."""
    model = GPT(tiny_cfg).eval()
    idx = torch.randint(50, (1, 16))
    altered = idx.clone()
    altered[0, 10:] = (altered[0, 10:] + 1) % 50
    a, _ = model(idx)
    b, _ = model(altered)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_initial_loss_is_random_guessing(tiny_cfg):
    model = GPT(tiny_cfg)
    idx = torch.randint(50, (8, 16))
    _, loss = model(idx, torch.randint(50, (8, 16)))
    assert abs(loss.item() - math.log(50)) < 0.3


def test_model_can_memorize_a_batch(tiny_cfg):
    """If the model + optimizer can't overfit one fixed batch, something is broken."""
    torch.manual_seed(0)
    model = GPT(tiny_cfg)
    params = list(model.parameters())
    opt = AdamW([{"params": params, "weight_decay": 0.0}])
    x = torch.randint(50, (4, 16))
    y = torch.randint(50, (4, 16))
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step(lr=3e-3)
    assert loss.item() < 0.5


def test_generate_shape(tiny_cfg):
    model = GPT(tiny_cfg).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 40, top_k=5)  # longer than block_size
    assert out.shape == (1, 41)


# --- optimizer ---------------------------------------------------------------

def test_adamw_matches_torch():
    torch.manual_seed(0)
    w_ours = torch.randn(10, 5, requires_grad=True)
    w_ref = w_ours.detach().clone().requires_grad_(True)
    ours = AdamW([{"params": [w_ours], "weight_decay": 0.1}])
    ref = torch.optim.AdamW([w_ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(10):
        target = torch.randn(10, 5)
        for w in (w_ours, w_ref):
            w.grad = None
            ((w - target) ** 2).sum().backward()
        ours.step(lr=1e-2)
        ref.step()
    assert torch.allclose(w_ours, w_ref, atol=1e-6)


def test_clip_grad_norm_matches_torch():
    a = [torch.randn(5, 5, requires_grad=True) for _ in range(3)]
    b = [p.detach().clone().requires_grad_(True) for p in a]
    for p, q in zip(a, b):
        g = torch.randn(5, 5) * 10
        p.grad, q.grad = g.clone(), g.clone()
    norm_ours = clip_grad_norm(a, 1.0)
    norm_ref = torch.nn.utils.clip_grad_norm_(b, 1.0)
    assert abs(norm_ours - norm_ref.item()) < 1e-3
    for p, q in zip(a, b):
        assert torch.allclose(p.grad, q.grad, atol=1e-5)
