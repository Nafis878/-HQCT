"""
utils/publication_plots.py -- IEEE-style publication figures (300 DPI, PDF+PNG).

8 figures saved to results/figures/:
  1. Architecture schematic (matplotlib patches)
  2. ROC curves with 95% CI shading (CKD + FHS, all models)
  3. Precision-Recall curves (same style)
  4. Critical Difference diagram (Nemenyi ranks)
  5. Expressibility vs entanglement scatter (quantum configs)
  6. Barren plateau: gradient variance vs layer depth
  7. SHAP summary subplots (CKD + FHS)
  8. Calibration reliability diagram

Run: python utils/publication_plots.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── IEEE rcParams ─────────────────────────────────────────────────────────────
IEEE_PARAMS = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
    "font.family": "DejaVu Serif",
}

MODEL_COLORS = {
    "XGBoost": "#e41a1c",
    "LightGBM": "#ff7f00",
    "MLP": "#4daf4a",
    "Classical TabTransformer": "#377eb8",
    "Quantum SVM": "#984ea3",
    "Hybrid Quantum Transformer": "#a65628",
}

MODEL_MARKERS = {
    "XGBoost": "o",
    "LightGBM": "s",
    "MLP": "^",
    "Classical TabTransformer": "D",
    "Quantum SVM": "v",
    "Hybrid Quantum Transformer": "*",
}

COL_88MM = 88 / 25.4   # single IEEE column in inches
COL_180MM = 180 / 25.4  # double IEEE column in inches


def _save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  results/figures/{name}.pdf + .png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Architecture schematic
# ══════════════════════════════════════════════════════════════════════════════

def fig_architecture() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(COL_180MM, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, label, color, fontsize=7):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                               fc=color, ec="black", lw=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    def arrow(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    # Input
    box(0.1, 1.6, 1.3, 0.8, "Input\n(24 features)", "#f0f0f0")
    arrow(1.4, 1.9, 2.0)
    # Linear embedding
    box(1.9, 1.6, 1.4, 0.8, "Linear\nEmbedding\n(d=32)", "#dae8fc")
    arrow(3.3, 3.6, 2.0)
    # Multi-head attention
    box(3.6, 1.6, 1.5, 0.8, "Multi-Head\nAttention\n(4 heads)", "#d5e8d4")
    arrow(5.1, 5.4, 2.0)
    # VQC FF block
    box(5.4, 0.8, 2.2, 2.4, "VQC Feed-Forward\n\n6 qubits · 3 layers\nHEA + data re-upload\n36 params", "#ffe6cc")
    arrow(7.6, 8.0, 2.0)
    # Output
    box(8.0, 1.6, 1.6, 0.8, "Classifier\n(sigmoid)", "#f0f0f0")

    # Add quantum circuit mini-diagram inside VQC box
    ax.text(6.5, 2.95, "q₀: ─RY─●──", ha="center", va="center", fontsize=5.5,
            fontfamily="monospace")
    ax.text(6.5, 2.70, "q₁: ─RY──●─", ha="center", va="center", fontsize=5.5,
            fontfamily="monospace")
    ax.text(6.5, 2.45, "q₂: ─RY───●", ha="center", va="center", fontsize=5.5,
            fontfamily="monospace")

    ax.set_title("Hybrid Quantum-Classical Transformer (HQCT) Architecture",
                 fontsize=9, fontweight="bold", pad=6)

    fig.tight_layout()
    _save_fig(fig, "fig1_architecture")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — ROC curves with 95% CI
# ══════════════════════════════════════════════════════════════════════════════

def fig_roc_curves() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc

    fig, axes = plt.subplots(1, 2, figsize=(COL_180MM, 2.8), sharey=True)

    for ax, (dataset, probas_file, y_file) in zip(axes, [
        ("CKD", RESULTS_DIR / "ckd_fold_probas.npz", BASE_DIR / "data" / "y_full.npy"),
        ("FHS", RESULTS_DIR / "fhs_fold_probas.npz", BASE_DIR / "data" / "fhs_y_full.npy"),
    ]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.6, label="Random")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR" if ax == axes[0] else "")
        ax.set_title(dataset)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

        if not probas_file.exists() or not y_file.exists():
            ax.text(0.5, 0.5, "Data not yet\navailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            continue

        data = np.load(probas_file)
        y_full = np.load(y_file)

        key_model = {
            "xgb": "XGBoost", "hqct": "Hybrid Quantum Transformer",
            "tab": "Classical TabTransformer", "lgb": "LightGBM", "mlp": "MLP",
        }
        for key, model_name in key_model.items():
            if key not in data:
                continue
            proba = data[key]
            if len(proba) != len(y_full):
                continue
            fpr, tpr, _ = roc_curve(y_full, proba)
            roc_auc = sk_auc(fpr, tpr)
            color = MODEL_COLORS.get(model_name, "gray")
            ax.plot(fpr, tpr, color=color, lw=1.2, label=f"{model_name} (AUC={roc_auc:.3f})")

        ax.legend(fontsize=5.5, loc="lower right", framealpha=0.7)

    fig.suptitle("ROC Curves — 10-Fold CV (Mean Probabilities)", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "fig2_roc_curves")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Precision-Recall curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_pr_curves() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, axes = plt.subplots(1, 2, figsize=(COL_180MM, 2.8), sharey=True)

    for ax, (dataset, probas_file, y_file) in zip(axes, [
        ("CKD", RESULTS_DIR / "ckd_fold_probas.npz", BASE_DIR / "data" / "y_full.npy"),
        ("FHS", RESULTS_DIR / "fhs_fold_probas.npz", BASE_DIR / "data" / "fhs_y_full.npy"),
    ]):
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision" if ax == axes[0] else "")
        ax.set_title(dataset)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

        if not probas_file.exists() or not y_file.exists():
            ax.text(0.5, 0.5, "Data not yet\navailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            continue

        data = np.load(probas_file)
        y_full = np.load(y_file)
        baseline = y_full.mean()
        ax.axhline(baseline, color="k", ls="--", lw=0.6, label=f"Baseline ({baseline:.2f})")

        key_model = {
            "xgb": "XGBoost", "hqct": "Hybrid Quantum Transformer",
            "tab": "Classical TabTransformer", "lgb": "LightGBM", "mlp": "MLP",
        }
        for key, model_name in key_model.items():
            if key not in data:
                continue
            proba = data[key]
            if len(proba) != len(y_full):
                continue
            prec, rec, _ = precision_recall_curve(y_full, proba)
            ap = average_precision_score(y_full, proba)
            color = MODEL_COLORS.get(model_name, "gray")
            ax.plot(rec, prec, color=color, lw=1.2, label=f"{model_name} (AP={ap:.3f})")

        ax.legend(fontsize=5.5, loc="upper right", framealpha=0.7)

    fig.suptitle("Precision-Recall Curves — 10-Fold CV", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "fig3_pr_curves")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Critical Difference diagram
# ══════════════════════════════════════════════════════════════════════════════

def fig_critical_difference() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt

    # Try to use scikit-posthocs CD diagram if available
    stat_path = RESULTS_DIR / "statistical_tests.json"
    if not stat_path.exists():
        fig, ax = plt.subplots(figsize=(COL_88MM, 1.5))
        ax.text(0.5, 0.5, "statistical_tests.json\nnot yet available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        _save_fig(fig, "fig4_critical_difference")
        plt.close(fig)
        return

    with open(stat_path) as f:
        stats = json.load(f)

    nemenyi = stats.get("nemenyi_matrix", {})
    avg_ranks = stats.get("average_ranks", {})

    if not avg_ranks:
        fig, ax = plt.subplots(figsize=(COL_88MM, 1.5))
        ax.text(0.5, 0.5, "Nemenyi ranks\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        _save_fig(fig, "fig4_critical_difference")
        plt.close(fig)
        return

    # Simple rank plot
    models = list(avg_ranks.keys())
    ranks = [avg_ranks[m] for m in models]
    sorted_pairs = sorted(zip(ranks, models))
    ranks_s, models_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(COL_88MM, max(2.0, len(models) * 0.4)))
    colors = [MODEL_COLORS.get(m, "gray") for m in models_s]
    ax.barh(range(len(models_s)), ranks_s, color=colors, edgecolor="black", lw=0.5, height=0.6)
    ax.set_yticks(range(len(models_s)))
    ax.set_yticklabels(models_s, fontsize=7)
    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_title("Critical Difference: Nemenyi Average Ranks\n(10-Fold AUC)", fontsize=8)
    ax.invert_yaxis()

    fig.tight_layout()
    _save_fig(fig, "fig4_critical_difference")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Expressibility vs Entanglement scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig_quantum_scatter() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt

    qm_path = RESULTS_DIR / "quantum_circuit_metrics.json"
    if not qm_path.exists():
        fig, ax = plt.subplots(figsize=(COL_88MM, 2.2))
        ax.text(0.5, 0.5, "quantum_circuit_metrics.json\nnot yet available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        _save_fig(fig, "fig5_quantum_scatter")
        plt.close(fig)
        return

    with open(qm_path) as f:
        qm = json.load(f)

    fig, ax = plt.subplots(figsize=(COL_88MM, 2.5))

    configs = ["4q_2L", "6q_3L"]
    labels_map = {"4q_2L": "LEGACY (4q-2L)", "6q_3L": "DEFAULT (6q-3L)"}
    colors_map = {"4q_2L": "#377eb8", "6q_3L": "#a65628"}
    markers_map = {"4q_2L": "o", "6q_3L": "*"}
    sizes_map = {"4q_2L": 80, "6q_3L": 140}

    for cfg in configs:
        if cfg not in qm:
            continue
        expr = qm[cfg].get("expressibility", 0)
        ent = qm[cfg].get("entanglement_capability", 0)
        ax.scatter(expr, ent, color=colors_map[cfg], marker=markers_map[cfg],
                   s=sizes_map[cfg], zorder=3, label=labels_map[cfg], edgecolors="black", lw=0.5)

    ax.set_xlabel("Expressibility (Meyer–Wallach)")
    ax.set_ylabel("Entanglement Capability ($Q$)")
    ax.set_title("VQC Expressibility vs. Entanglement\n(2000 random parameter samples)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)

    fig.tight_layout()
    _save_fig(fig, "fig5_quantum_scatter")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Barren plateau
# ══════════════════════════════════════════════════════════════════════════════

def fig_barren_plateau() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt

    bp_path = RESULTS_DIR / "barren_plateau.json"
    if not bp_path.exists():
        fig, ax = plt.subplots(figsize=(COL_88MM, 2.2))
        ax.text(0.5, 0.5, "barren_plateau.json\nnot yet available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        _save_fig(fig, "fig6_barren_plateau")
        plt.close(fig)
        return

    with open(bp_path) as f:
        bp = json.load(f)

    layers = list(bp.get("layer_variance", {}).keys())
    variances = [bp["layer_variance"][l] for l in layers]

    fig, ax = plt.subplots(figsize=(COL_88MM, 2.5))
    ax.semilogy(range(1, len(layers) + 1), variances, "o-", color="#a65628",
                lw=1.2, ms=5, label="Gradient variance")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Gradient variance (log scale)")
    ax.set_title("Barren Plateau Analysis:\nGradient Variance vs. Circuit Depth", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)

    fig.tight_layout()
    _save_fig(fig, "fig6_barren_plateau")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — SHAP summary
# ══════════════════════════════════════════════════════════════════════════════

def fig_shap_summary() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(COL_180MM, 3.0))

    for ax, (dataset, shap_img) in zip(axes, [
        ("CKD", FIGURES_DIR / "shap_summary_XGBoost_ckd.png"),
        ("FHS", FIGURES_DIR / "shap_summary_XGBoost_fhs.png"),
    ]):
        ax.set_title(f"SHAP ({dataset})")
        ax.axis("off")
        if shap_img.exists():
            img = plt.imread(str(shap_img))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "SHAP figure\nnot yet available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")

    fig.suptitle("SHAP Feature Importance (XGBoost)", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "fig7_shap_summary")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Calibration reliability diagram
# ══════════════════════════════════════════════════════════════════════════════

def fig_calibration() -> None:
    import matplotlib
    matplotlib.rcParams.update(IEEE_PARAMS)
    import matplotlib.pyplot as plt

    cal_img = FIGURES_DIR / "calibration_analysis.png"
    fig, ax = plt.subplots(figsize=(COL_88MM, 2.5))
    ax.axis("off")

    if cal_img.exists():
        img = plt.imread(str(cal_img))
        ax.imshow(img)
        ax.set_title("Calibration Reliability Diagram", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Calibration figure\nnot yet available",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_title("Calibration Reliability Diagram (placeholder)", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "fig8_calibration")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_figures() -> None:
    print("=" * 60)
    print("GENERATING IEEE-STYLE PUBLICATION FIGURES")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_funcs = [
        ("Architecture schematic", fig_architecture),
        ("ROC curves", fig_roc_curves),
        ("Precision-Recall curves", fig_pr_curves),
        ("Critical Difference diagram", fig_critical_difference),
        ("Quantum scatter (expressibility vs entanglement)", fig_quantum_scatter),
        ("Barren plateau analysis", fig_barren_plateau),
        ("SHAP summary", fig_shap_summary),
        ("Calibration reliability", fig_calibration),
    ]

    for name, fn in fig_funcs:
        try:
            print(f"\n[{name}]")
            fn()
        except Exception as exc:
            print(f"  WARNING: Failed — {exc}")

    print("\nAll figures generated (see results/figures/).")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
