"""
utils/quantum_advantage.py -- Quantum advantage analysis tools.

Implements:
  - Kernel Target Alignment (KTA): compares VQC implicit kernel to RBF kernel.
  - Barren plateau check: gradient variance vs circuit depth.
  - Effective dimension: Fisher Information Matrix rank as capacity proxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Kernel Target Alignment ───────────────────────────────────────────────────

def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center kernel matrix: Kc = K - 1_n K - K 1_n + 1_n K 1_n"""
    n = K.shape[0]
    ones = np.ones((n, n)) / n
    return K - ones @ K - K @ ones + ones @ K @ ones


def kernel_target_alignment(
    X: np.ndarray,
    y: np.ndarray,
    quantum_kernel_fn: Callable,
    classical_kernel: str = "rbf",
    n_samples: int = 100,
    rng_seed: int = 42,
) -> dict:
    """
    Kernel Target Alignment (KTA) score comparing the VQC's implicit kernel
    to a classical RBF kernel on the same data.

    KTA(K1, K2) = <K1, K2>_F / (||K1||_F * ||K2||_F)

    Returns KTA scores for both kernels vs the ideal label kernel.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    rng = np.random.RandomState(rng_seed)
    n = min(n_samples, len(X))
    idx = rng.choice(len(X), n, replace=False)
    Xs, ys = X[idx], y[idx]

    # Ideal kernel: yy^T
    y_col = ys.reshape(-1, 1).astype(float) * 2 - 1  # {-1, +1}
    K_ideal = y_col @ y_col.T

    # Classical RBF kernel
    K_rbf = rbf_kernel(Xs)

    # Quantum kernel (may be slow)
    try:
        K_q = quantum_kernel_fn(Xs)
    except Exception as e:
        return {"error": f"quantum_kernel_fn failed: {e}"}

    def kta(Ka, Kb):
        Ka_c = _center_kernel(Ka)
        Kb_c = _center_kernel(Kb)
        num = float(np.sum(Ka_c * Kb_c))
        den = float(np.linalg.norm(Ka_c, "fro") * np.linalg.norm(Kb_c, "fro") + 1e-12)
        return num / den

    return {
        "kta_quantum_vs_ideal": kta(K_q, K_ideal),
        "kta_rbf_vs_ideal": kta(K_rbf, K_ideal),
        "kta_quantum_vs_rbf": kta(K_q, K_rbf),
        "n_samples_used": n,
    }


# ── Barren plateau check ──────────────────────────────────────────────────────

def barren_plateau_check(
    model,
    n_samples: int = 300,
    out_path: Optional[str] = None,
    rng_seed: int = 42,
) -> dict:
    """
    Compute gradient variance w.r.t. each variational parameter of the quantum layer.
    Plots gradient variance vs parameter index; saves to out_path if given.

    A barren plateau manifests as exponentially vanishing variance with circuit depth.
    Returns dict with per-parameter gradient variance.
    """
    import torch

    rng = np.random.RandomState(rng_seed)
    model.eval()

    # Collect quantum parameters
    q_params = [p for name, p in model.named_parameters() if "weight" in name or "theta" in name or "params" in name]
    if not q_params:
        q_params = list(model.parameters())

    grad_vars = []
    for param in q_params:
        grads = []
        for _ in range(n_samples):
            model.zero_grad()
            # Random input
            n_feat = getattr(model, "n_features", 24)
            x = torch.randn(1, n_feat)
            try:
                out = model(x)
                loss = out.sum()
                loss.backward()
                if param.grad is not None:
                    grads.append(param.grad.detach().cpu().numpy().flatten())
            except Exception:
                break

        if grads:
            grads_arr = np.array(grads)
            grad_vars.append(float(np.var(grads_arr)))
        else:
            grad_vars.append(0.0)

    if out_path and grad_vars:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(range(len(grad_vars)), grad_vars, "o-", color="#00D9FF", lw=2, ms=5)
        ax.set_xlabel("Variational Parameter Index", fontsize=11)
        ax.set_ylabel("Gradient Variance (log scale)", fontsize=11)
        ax.set_title("Barren Plateau Analysis — Gradient Variance vs Parameter Index",
                     fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [quantum_advantage] Barren plateau plot saved: {out_path}")

    return {
        "n_params_checked": len(grad_vars),
        "mean_gradient_variance": float(np.mean(grad_vars)) if grad_vars else 0.0,
        "min_gradient_variance": float(np.min(grad_vars)) if grad_vars else 0.0,
        "max_gradient_variance": float(np.max(grad_vars)) if grad_vars else 0.0,
        "gradient_variances": grad_vars,
        "plateau_detected": bool(np.mean(grad_vars) < 1e-6) if grad_vars else False,
    }


# ── Effective dimension ───────────────────────────────────────────────────────

def effective_dimension(
    model,
    X: np.ndarray,
    n_samples: int = 200,
    rng_seed: int = 42,
) -> dict:
    """
    Estimate effective model dimension via Fisher Information Matrix (FIM) rank.
    Higher rank = richer parameter space = more capacity.
    """
    import torch

    rng = np.random.RandomState(rng_seed)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    gradients = []
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)

    for i in idx:
        model.zero_grad()
        x = torch.tensor(X[i : i + 1], dtype=torch.float32)
        try:
            out = model(x)
            out.sum().backward()
            grad = np.concatenate([
                p.grad.detach().cpu().numpy().flatten()
                for p in params if p.grad is not None
            ])
            gradients.append(grad)
        except Exception:
            continue

    if not gradients:
        return {"error": "Could not compute gradients", "n_params": n_params}

    G = np.array(gradients)   # shape: (n_samples, n_params)
    FIM = G.T @ G / len(gradients)

    singular_values = np.linalg.svd(FIM, compute_uv=False)
    # Effective rank via normalized entropy of singular values
    sv_norm = singular_values / (singular_values.sum() + 1e-12)
    sv_norm = sv_norm[sv_norm > 1e-10]
    eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-12))))

    return {
        "n_params_total": int(n_params),
        "fim_effective_rank": float(eff_rank),
        "top5_singular_values": singular_values[:5].tolist(),
        "n_samples_used": len(gradients),
    }


def save_quantum_advantage(results: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    print(f"  [quantum_advantage] Saved: {out_path}")
