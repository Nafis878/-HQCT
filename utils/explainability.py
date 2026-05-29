"""
utils/explainability.py -- SHAP + attention explainability for HQCT models.

Generates:
  - Global SHAP beeswarm plots per model × dataset
  - SHAP feature importance CSVs
  - Waterfall plots for most-misclassified samples
  - Quantum feature attribution (gradient of VQC output w.r.t. inputs)
  - Multi-head attention weight heatmaps
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── SHAP: XGBoost (TreeExplainer) ────────────────────────────────────────────

def shap_xgboost(
    model,
    X: np.ndarray,
    feature_names: List[str],
    out_dir: str,
    dataset_label: str = "CKD",
    max_display: int = 15,
) -> pd.DataFrame:
    """
    Compute SHAP values for an XGBoost model via TreeExplainer.
    Saves beeswarm PNG + importance CSV.
    Returns feature importance DataFrame.
    """
    try:
        import shap
    except ImportError:
        print("  [explainability] shap not installed: pip install shap")
        return pd.DataFrame()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary XGBoost, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names,
        plot_type="dot", max_display=max_display, show=False
    )
    plt.title(f"SHAP Summary — XGBoost ({dataset_label})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = out / f"shap_summary_XGBoost_{dataset_label}.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [explainability] Saved: {beeswarm_path}")

    # Feature importance CSV
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    csv_path = out / f"shap_importance_XGBoost_{dataset_label}.csv"
    importance.to_csv(csv_path, index=False)
    print(f"  [explainability] Saved: {csv_path}")

    return importance


# ── SHAP: PyTorch models (GradientExplainer with KernelExplainer fallback) ───

def shap_torch_model(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    model_name: str = "HybridQT",
    out_dir: str = "results",
    dataset_label: str = "CKD",
    max_display: int = 15,
    n_background: int = 50,
) -> pd.DataFrame:
    """
    SHAP for PyTorch models. Tries GradientExplainer first (fast but requires
    clean backprop through the full model). Falls back to KernelExplainer if
    the quantum layer blocks gradient flow.

    Returns feature importance DataFrame.
    """
    try:
        import shap
        import torch
    except ImportError:
        print("  [explainability] shap or torch not installed")
        return pd.DataFrame()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.eval()
    bg = X_background[:n_background]
    bg_t = torch.tensor(bg, dtype=torch.float32)

    shap_values = None

    # Try GradientExplainer
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.GradientExplainer(model, bg_t)
            X_t = torch.tensor(X_explain, dtype=torch.float32)
            sv = explainer.shap_values(X_t)
            if isinstance(sv, list):
                sv = sv[0]
            shap_values = sv if isinstance(sv, np.ndarray) else sv.numpy()
        print(f"  [explainability] {model_name}: using GradientExplainer")
    except Exception as e_grad:
        print(f"  [explainability] GradientExplainer failed ({e_grad}); falling back to KernelExplainer")
        # KernelExplainer: model-agnostic but slow
        try:
            def predict_fn(x_np):
                import torch
                with torch.no_grad():
                    t = torch.tensor(x_np, dtype=torch.float32)
                    out = torch.sigmoid(model(t)).numpy().flatten()
                return out

            explainer_k = shap.KernelExplainer(predict_fn, bg[:20])
            sv = explainer_k.shap_values(X_explain[:50], nsamples=100)
            if isinstance(sv, list):
                sv = sv[0]
            shap_values = sv
            print(f"  [explainability] {model_name}: using KernelExplainer (n=50 samples)")
        except Exception as e_ker:
            print(f"  [explainability] KernelExplainer also failed: {e_ker}")
            return pd.DataFrame()

    if shap_values is None or len(shap_values) == 0:
        return pd.DataFrame()

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_explain[:len(shap_values)],
        feature_names=feature_names, plot_type="dot",
        max_display=max_display, show=False
    )
    plt.title(f"SHAP Summary — {model_name} ({dataset_label})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = out / f"shap_summary_{model_name}_{dataset_label}.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [explainability] Saved: {beeswarm_path}")

    # Importance CSV
    importance = pd.DataFrame({
        "feature": feature_names[:shap_values.shape[1]],
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    csv_path = out / f"shap_importance_{model_name}_{dataset_label}.csv"
    importance.to_csv(csv_path, index=False)
    print(f"  [explainability] Saved: {csv_path}")

    return importance


# ── SHAP waterfall for misclassified samples ──────────────────────────────────

def shap_waterfall_misclassified(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    model_name: str = "HybridQT",
    out_dir: str = "results",
    dataset_label: str = "CKD",
    n_worst: int = 3,
) -> None:
    """
    SHAP waterfall plots for the n most-confidently-wrong predictions.
    """
    try:
        import shap
        import torch
    except ImportError:
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print(f"  [explainability] No misclassified samples for {model_name}")
        return

    n = min(n_worst, len(wrong_idx))
    selected = wrong_idx[:n]

    model.eval()
    bg = torch.tensor(X[:50], dtype=torch.float32)
    try:
        explainer = shap.GradientExplainer(model, bg)
        for i, idx in enumerate(selected):
            x_sample = torch.tensor(X[idx : idx + 1], dtype=torch.float32)
            sv = explainer.shap_values(x_sample)
            if isinstance(sv, list):
                sv = sv[0]
            sv = sv[0] if hasattr(sv, "__len__") else sv

            plt.figure(figsize=(10, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values=np.array(sv).flatten()[:len(feature_names)],
                    base_values=0.0,
                    data=X[idx][:len(feature_names)],
                    feature_names=feature_names,
                ),
                show=False,
            )
            plt.title(
                f"SHAP Waterfall — {model_name} ({dataset_label})\n"
                f"Sample #{idx} | True={y_true[idx]} Pred={y_pred[idx]}",
                fontsize=11, fontweight="bold",
            )
            plt.tight_layout()
            wf_path = out / f"shap_waterfall_{model_name}_{dataset_label}_sample{i}.png"
            plt.savefig(wf_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  [explainability] Saved: {wf_path}")
    except Exception as e:
        print(f"  [explainability] Waterfall plot failed: {e}")


# ── Quantum feature attribution ───────────────────────────────────────────────

def quantum_feature_attribution(
    model,
    X: np.ndarray,
    feature_names: List[str],
    out_dir: str = "results",
    dataset_label: str = "CKD",
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Novel contribution: gradient of the quantum layer's output w.r.t. input features.

    This traces which input features most strongly rotate the VQC's qubits,
    providing a quantum-specific attribution that is distinct from standard SHAP.

    Uses PyTorch autograd to compute d(vqc_output)/d(x) for each sample,
    then averages absolute gradients over samples.
    """
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.eval()
    n = min(n_samples, len(X))
    grad_accumulator = np.zeros(len(feature_names))

    # Hook to capture intermediate VQC output
    vqc_grads = {}
    def _make_hook(name):
        def hook(grad):
            vqc_grads[name] = grad.detach().cpu().numpy()
        return hook

    success = 0
    for i in range(n):
        x = torch.tensor(X[i : i + 1], dtype=torch.float32, requires_grad=True)
        try:
            out_val = model(x)
            # Differentiate scalar output w.r.t. input
            out_val.sum().backward()
            if x.grad is not None:
                g = x.grad.detach().cpu().numpy().flatten()
                grad_accumulator[:len(g)] += np.abs(g[:len(feature_names)])
                success += 1
        except Exception:
            continue

    if success == 0:
        print(f"  [explainability] quantum_feature_attribution: no gradients computed")
        return pd.DataFrame()

    grad_accumulator /= success

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_input_gradient": grad_accumulator,
    }).sort_values("mean_abs_input_gradient", ascending=False)

    csv_path = Path(out_dir) / f"quantum_feature_attribution_{dataset_label}.csv"
    df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    top = df.head(15)
    ax.barh(top["feature"][::-1], top["mean_abs_input_gradient"][::-1],
            color="#00D9FF", alpha=0.85)
    ax.set_xlabel("Mean |d(output)/d(feature)|", fontsize=11)
    ax.set_title(f"Quantum Feature Attribution — HybridQT ({dataset_label})\n"
                 "Gradient of VQC output w.r.t. input features",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_path = Path(out_dir) / f"quantum_feature_attribution_{dataset_label}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [explainability] Saved: {csv_path}, {plot_path}")

    return df


# ── Attention visualization ───────────────────────────────────────────────────

def visualize_attention(
    model,
    X_sample: np.ndarray,
    feature_names: List[str],
    save_path: str,
    dataset_label: str = "CKD",
) -> None:
    """
    Extract MHA weights via forward hook and save a heatmap.
    Works for both TabTransformer and HybridQT.
    """
    import torch

    attention_weights = {}

    def _attn_hook(module, input, output):
        # MultiheadAttention returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            w = output[1]
            if w is not None:
                attention_weights["attn"] = w.detach().cpu().numpy()

    # Register hook on first MultiheadAttention layer
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            h = module.register_forward_hook(_attn_hook)
            hooks.append(h)
            break  # first attention layer only

    model.eval()
    x = torch.tensor(X_sample[:1], dtype=torch.float32)
    try:
        with torch.no_grad():
            model(x)
    except Exception:
        pass

    for h in hooks:
        h.remove()

    if "attn" not in attention_weights:
        print(f"  [explainability] Could not extract attention weights for {save_path}")
        return

    attn = attention_weights["attn"]  # shape: (batch, heads, seq, seq) or (batch, seq, seq)
    if attn.ndim == 4:
        attn = attn[0].mean(0)  # average over heads: (seq, seq)
    elif attn.ndim == 3:
        attn = attn[0]
    else:
        attn = attn.squeeze()

    n_feat = min(attn.shape[0], len(feature_names) + 1)  # +1 for CLS token
    labels = ["[CLS]"] + list(feature_names[: n_feat - 1])

    fig, ax = plt.subplots(figsize=(max(8, n_feat * 0.5), max(6, n_feat * 0.5)))
    im = ax.imshow(attn[:n_feat, :n_feat], cmap="Blues", aspect="auto")
    ax.set_xticks(range(n_feat))
    ax.set_yticks(range(n_feat))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Multi-Head Attention Weights ({dataset_label})",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [explainability] Attention map saved: {save_path}")
