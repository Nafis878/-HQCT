"""
combined_plots.py -- Publication-quality dual-dataset comparison plots.
Generates 3 figures (dpi=300) comparing CKD and FHS results.
Step FHS-5 of the QIP 2027 dual-dataset pipeline.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

CKD_CSV = RESULTS_DIR / "cv_results.csv"
FHS_CSV = RESULTS_DIR / "fhs_cv_results.csv"
CKD_PROBAS = RESULTS_DIR / "ckd_fold_probas.npz"
FHS_PROBAS = RESULTS_DIR / "fhs_fold_probas.npz"

MODEL_ORDER = ["XGBoost", "QSVM", "Classical TabTransformer", "Hybrid Quantum Transformer"]
DISPLAY_NAMES = {
    "XGBoost": "XGBoost",
    "QSVM": "QSVM",
    "Classical TabTransformer": "TabTransformer",
    "Hybrid Quantum Transformer": "HybridQT",
}


def _align_models(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows to match MODEL_ORDER; skip missing models."""
    ordered = []
    for m in MODEL_ORDER:
        row = df[df["Model"].str.contains(m.split()[0], case=False, na=False)]
        if len(row) > 0:
            ordered.append(row.iloc[0])
    return pd.DataFrame(ordered).reset_index(drop=True)


def plot_accuracy_comparison(df_ckd: pd.DataFrame, df_fhs: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: CKD vs FHS accuracy per model."""
    df_ckd_a = _align_models(df_ckd)
    df_fhs_a = _align_models(df_fhs)

    # Use intersection of available models
    ckd_models = list(df_ckd_a["Model"])
    fhs_models = list(df_fhs_a["Model"])

    # Build label lookup
    def short(name):
        for k, v in DISPLAY_NAMES.items():
            if k.split()[0].lower() in name.lower():
                return v
        return name

    ckd_labels = [short(m) for m in ckd_models]
    fhs_labels = [short(m) for m in fhs_models]

    # Union of models
    all_labels = list(dict.fromkeys(ckd_labels + fhs_labels))
    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ckd_acc = [df_ckd_a[df_ckd_a["Model"].apply(short) == lbl]["Accuracy"].values[0] * 100
               if lbl in ckd_labels else 0 for lbl in all_labels]
    ckd_std = [df_ckd_a[df_ckd_a["Model"].apply(short) == lbl]["Accuracy_std"].values[0] * 100
               if lbl in ckd_labels else 0 for lbl in all_labels]
    fhs_acc = [df_fhs_a[df_fhs_a["Model"].apply(short) == lbl]["Accuracy"].values[0] * 100
               if lbl in fhs_labels else 0 for lbl in all_labels]
    fhs_std = [df_fhs_a[df_fhs_a["Model"].apply(short) == lbl]["Accuracy_std"].values[0] * 100
               if lbl in fhs_labels else 0 for lbl in all_labels]

    bars1 = ax.bar(x - width/2, ckd_acc, width, yerr=ckd_std, capsize=5,
                   color="#2196F3", alpha=0.85, label="CKD Dataset")
    bars2 = ax.bar(x + width/2, fhs_acc, width, yerr=fhs_std, capsize=5,
                   color="#FF9800", alpha=0.85, label="FHS Dataset")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("10-Fold CV Accuracy: CKD vs Framingham Heart Study", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=11)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, val in zip(bars1, ckd_acc):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, fhs_acc):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_variance_comparison(df_ckd: pd.DataFrame, df_fhs: pd.DataFrame, out: Path) -> None:
    """Std dev comparison: highlights HQCT lower variance."""
    df_ckd_a = _align_models(df_ckd)
    df_fhs_a = _align_models(df_fhs)

    def short(name):
        for k, v in DISPLAY_NAMES.items():
            if k.split()[0].lower() in name.lower():
                return v
        return name

    ckd_labels = [short(m) for m in df_ckd_a["Model"]]
    fhs_labels = [short(m) for m in df_fhs_a["Model"]]
    all_labels = list(dict.fromkeys(ckd_labels + fhs_labels))
    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ckd_std = [df_ckd_a[df_ckd_a["Model"].apply(short) == lbl]["Accuracy_std"].values[0] * 100
               if lbl in ckd_labels else 0 for lbl in all_labels]
    fhs_std = [df_fhs_a[df_fhs_a["Model"].apply(short) == lbl]["Accuracy_std"].values[0] * 100
               if lbl in fhs_labels else 0 for lbl in all_labels]

    # Color HQCT bars green to highlight quantum advantage
    ckd_colors = ["#4CAF50" if lbl == "HybridQT" else "#2196F3" for lbl in all_labels]
    fhs_colors = ["#4CAF50" if lbl == "HybridQT" else "#FF9800" for lbl in all_labels]

    ax.bar(x - width/2, ckd_std, width, color=ckd_colors, alpha=0.85, label="CKD Dataset")
    ax.bar(x + width/2, fhs_std, width, color=fhs_colors, alpha=0.85, label="FHS Dataset")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy Std Dev (%)", fontsize=12)
    ax.set_title("Cross-Validation Stability: Quantum vs Classical Models\n"
                 "(Lower = More Stable; Green = HybridQT)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=11)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.85, label="CKD (classical)"),
        Patch(facecolor="#FF9800", alpha=0.85, label="FHS (classical)"),
        Patch(facecolor="#4CAF50", alpha=0.85, label="HybridQT (both datasets)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _mean_roc(fold_probas: np.ndarray, fold_labels: np.ndarray):
    """Compute mean ROC curve with std band across folds."""
    base_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    for i in range(fold_probas.shape[0]):
        mask = ~np.isnan(fold_probas[i])
        if mask.sum() < 2:
            continue
        proba = fold_probas[i][mask]
        labels = fold_labels[i][mask].astype(int)
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, proba)
        tprs.append(np.interp(base_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    return base_fpr, mean_tpr, std_tpr, mean_auc, std_auc


def _roc_panel(ax, data, proba_key, label_key, title, color) -> None:
    """Draw one ROC panel onto ax."""
    if proba_key not in data or label_key not in data:
        ax.set_title(f"{title}\n(data not available)")
        ax.text(0.5, 0.5, "Skipped", ha="center", va="center", transform=ax.transAxes)
        return
    base_fpr, mean_tpr, std_tpr, mean_auc, std_auc = _mean_roc(
        data[proba_key], data[label_key]
    )
    ax.plot(base_fpr, mean_tpr, color=color, lw=2,
            label=f"Mean ROC (AUC = {mean_auc:.4f} +/- {std_auc:.4f})")
    ax.fill_between(base_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.2, color=color, label="+/- 1 std")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_roc_curves(out: Path) -> None:
    """ROC curves: 2x2 if both proba files exist, 1x2 (FHS only) otherwise."""
    ckd_ok = CKD_PROBAS.exists()
    fhs_ok = FHS_PROBAS.exists()

    if not ckd_ok and not fhs_ok:
        print(f"  Skipping ROC plot: neither proba file found.")
        print(f"    Expected: {CKD_PROBAS}")
        print(f"    Expected: {FHS_PROBAS}")
        return

    if not ckd_ok:
        print(f"  Note: {CKD_PROBAS.name} not found — plotting FHS-only ROC (1x2 layout).")

    fhs = np.load(FHS_PROBAS) if fhs_ok else None
    ckd = np.load(CKD_PROBAS) if ckd_ok else None

    if ckd_ok and fhs_ok:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Mean ROC Curves with +/- 1 Std Band\n10-Fold Cross-Validation",
                     fontsize=13, fontweight="bold")
        _roc_panel(axes[0, 0], ckd, "xgb_probas", "xgb_labels", "CKD -- XGBoost", "#2196F3")
        _roc_panel(axes[0, 1], ckd, "hqct_probas", "hqct_labels", "CKD -- HybridQT", "#4CAF50")
        _roc_panel(axes[1, 0], fhs, "xgb_probas", "xgb_labels", "FHS -- XGBoost", "#FF9800")
        _roc_panel(axes[1, 1], fhs, "hqct_probas", "hqct_labels", "FHS -- HybridQT", "#9C27B0")
    else:
        # FHS-only 1x2 layout
        data = fhs if fhs_ok else ckd
        prefix = "FHS" if fhs_ok else "CKD"
        colors = ("#FF9800", "#9C27B0") if fhs_ok else ("#2196F3", "#4CAF50")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Mean ROC Curves with +/- 1 Std Band\n10-Fold CV ({prefix} Dataset)",
                     fontsize=13, fontweight="bold")
        _roc_panel(axes[0], data, "xgb_probas", "xgb_labels",
                   f"{prefix} -- XGBoost", colors[0])
        _roc_panel(axes[1], data, "hqct_probas", "hqct_labels",
                   f"{prefix} -- HybridQT", colors[1])

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    print("=" * 60)
    print("FHS STEP 5 -- COMBINED PLOTS")
    print("=" * 60)

    for path in [CKD_CSV, FHS_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                "Run cv_evaluation.py (CKD) and fhs_cv_evaluation.py (FHS) first."
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_ckd = pd.read_csv(CKD_CSV)
    df_fhs = pd.read_csv(FHS_CSV)

    print("\nGenerating Plot 1: Accuracy comparison...")
    plot_accuracy_comparison(df_ckd, df_fhs, RESULTS_DIR / "dual_accuracy_comparison.png")

    print("Generating Plot 2: Variance comparison...")
    plot_variance_comparison(df_ckd, df_fhs, RESULTS_DIR / "dual_variance_comparison.png")

    print("Generating Plot 3: ROC curves...")
    plot_roc_curves(RESULTS_DIR / "dual_roc_curves.png")

    print("\n" + "=" * 60)
    print("Combined plots complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
