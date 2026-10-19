"""
The optimizer: how the model actually learns.

After loss.backward(), every parameter p has p.grad, the direction that would
*increase* the loss fastest. The simplest optimizer (SGD) just steps the other
way: p -= lr * grad. AdamW is smarter in three ways:

  1. Momentum (m): a running average of recent gradients, so noisy batches
     don't jerk the weights around.
  2. Per-parameter step size (v): a running average of squared gradients.
     Dividing by sqrt(v) means parameters with consistently large gradients
     take smaller steps and rarely-updated ones take bigger steps.
  3. Weight decay ("W"): shrink weights slightly toward zero each step, which
     discourages the model from relying on a few huge weights.
"""

import math

import torch


class AdamW:
    def __init__(self, param_groups: list[dict], betas=(0.9, 0.95), eps: float = 1e-8):
        # param_groups: [{"params": [...], "weight_decay": 0.1}, ...]
        self.param_groups = param_groups
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0       # step counter, used for bias correction
        self.state = {}  # param -> (m, v)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self, lr: float):
        self.t += 1
        # m and v start at zero, so early on they underestimate the true
        # averages. Dividing by (1 - beta^t) corrects for that.
        bias_corr1 = 1 - self.beta1 ** self.t
        bias_corr2 = 1 - self.beta2 ** self.t

        for group in self.param_groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p] = (torch.zeros_like(p), torch.zeros_like(p))
                m, v = self.state[p]
                g = p.grad

                m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)         # m = b1*m + (1-b1)*g
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)  # v = b2*v + (1-b2)*g^2

                p.mul_(1 - lr * wd)  # weight decay, applied directly to the weights
                m_hat = m / bias_corr1
                v_hat = v / bias_corr2
                p.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=-lr)


@torch.no_grad()
def clip_grad_norm(params: list[torch.Tensor], max_norm: float) -> float:
    """If the gradient vector (all parameters together) is longer than max_norm,
    scale it down to max_norm. Stops a single bad batch from causing a huge
    update that wrecks training. Returns the norm before clipping."""
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(g.pow(2).sum().item() for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total
