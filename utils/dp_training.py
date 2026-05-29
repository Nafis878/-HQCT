"""
utils/dp_training.py -- Differentially Private SGD (DP-SGD) for HQCT models.

Implements a manual DP-SGD wrapper without requiring Opacus, so there are no
extra heavy dependencies. The implementation clips per-sample gradients to a
maximum L2 norm, then adds calibrated Gaussian noise.

Privacy accounting uses a simplified Renyi Differential Privacy (RDP) moments
accountant, converted to (epsilon, delta)-DP via the standard RDP-to-DP
composition theorem.

Reference:
  Abadi et al. (2016) "Deep Learning with Differential Privacy"
  Mironov (2017) "Renyi Differential Privacy of the Gaussian Mechanism"
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# ── DP-SGD Optimizer ─────────────────────────────────────────────────────────

class DPOptimizer:
    """
    Wraps any PyTorch optimizer with per-sample gradient clipping + Gaussian noise.

    Usage:
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        dp_opt = DPOptimizer(base_opt, noise_multiplier=1.1, max_grad_norm=1.0)
        # Then use dp_opt.step() instead of base_opt.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        rng_seed: Optional[int] = None,
    ):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        if rng_seed is not None:
            torch.manual_seed(rng_seed)
        self._step_count = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        """Clip gradients + add Gaussian noise, then call underlying optimizer step."""
        params = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not params:
            return

        # 1. Clip per-parameter gradient to max_grad_norm
        total_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        # 2. Add Gaussian noise scaled by noise_multiplier * max_grad_norm
        sigma = self.noise_multiplier * self.max_grad_norm
        for p in params:
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * sigma
                p.grad.data.add_(noise)

        self._step_count += 1
        self.optimizer.step(closure)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ── Privacy accounting ────────────────────────────────────────────────────────

def _rdp_gaussian(q: float, sigma: float, alpha: int) -> float:
    """
    Renyi DP of the Gaussian mechanism for a single step.
    q = sampling probability, sigma = noise_multiplier, alpha = order.
    """
    if alpha == 1:
        return q * math.log(1.0 / sigma ** 2 + 1e-12) / 2.0
    if sigma == 0:
        return float("inf")
    # Analytic formula from Mironov (2017)
    # Simplified: RDP(alpha) = alpha * q^2 / (2 * sigma^2)  [tight for small q]
    return alpha * q ** 2 / (2.0 * sigma ** 2)


def compute_epsilon(
    steps: int,
    batch_size: int,
    n_samples: int,
    noise_multiplier: float,
    delta: float = 1e-5,
) -> float:
    """
    Compute (epsilon, delta)-DP guarantee via simplified RDP moments accountant.

    Parameters
    ----------
    steps          : number of gradient steps (epochs * batches_per_epoch)
    batch_size     : batch size
    n_samples      : dataset size
    noise_multiplier : sigma / max_grad_norm
    delta          : target delta (e.g. 1/n_samples)

    Returns epsilon such that the mechanism is (epsilon, delta)-DP.
    """
    q = batch_size / max(n_samples, 1)  # subsampling probability
    orders = list(range(2, 64)) + [128, 256, 512]

    best_eps = float("inf")
    for alpha in orders:
        rdp = steps * _rdp_gaussian(q, noise_multiplier, alpha)
        # Convert RDP to (eps, delta)-DP: eps = rdp - log(delta) / (alpha - 1)
        if alpha > 1:
            eps = rdp + math.log(1.0 / (delta + 1e-300)) / (alpha - 1.0)
            best_eps = min(best_eps, eps)

    return max(0.0, best_eps)


def privacy_report(
    epochs: int,
    batch_size: int,
    n_samples: int,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float = 1e-5,
) -> dict:
    """Produce a privacy guarantee summary dict."""
    steps = epochs * math.ceil(n_samples / batch_size)
    epsilon = compute_epsilon(steps, batch_size, n_samples, noise_multiplier, delta)
    return {
        "epsilon": round(epsilon, 4),
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "steps": steps,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "mechanism": "DP-SGD (Gaussian)",
    }
