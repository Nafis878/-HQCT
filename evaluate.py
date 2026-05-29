"""
evaluate.py — Load all four trained models, compute metrics on the test set,
print formatted report blocks and a comparison table, and save publication-quality plots.

Models evaluated:
  1. XGBoost                   (sklearn/joblib)
  2. Quantum SVM (QSVM)        (sklearn/joblib + quantum kernel)
  3. Classical TabTransformer  (PyTorch .pt checkpoint)
  4. Hybrid Quantum Transformer (PyTorch .pt checkpoint + PennyLane VQC)

Step 5 of the QIP 2027 pipeline.
"""

import random
import sys
import warnings
from pathlib import Path

try:
    from utils.integrity import compute_sha256
    from utils.calibration import compute_calibration_metrics, reliability_diagram, save_calibration_metrics
    _HAS_Q1 = True
except ImportError:
    _HAS_Q1 = False

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Plot style ──────────────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
MODEL_NAMES = [
    "XGBoost",
    "Quantum SVM",
    "Classical TabTransformer",
    "Hybrid Quantum Transformer",
]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return a dict of classification metrics."""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="binary", zero_division=0),
        "F1":        f1_score(y_true, y_pred, average="binary", zero_division=0),
        "ROC_AUC":   roc_auc_score(y_true, y_proba),
    }


def print_metrics_block(model_name: str, metrics: dict, cm: np.ndarray) -> None:
    """Print a formatted metrics block for one model."""
    sep = "=" * 50
    dash = "-" * 50
    print(f"\n{sep}")
    print(f"  Model: {model_name}")
    print(f"{sep}")
    print(f"  Accuracy  : {metrics['Accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['Precision']*100:.2f}%")
    print(f"  Recall    : {metrics['Recall']*100:.2f}%")
    print(f"  F1-Score  : {metrics['F1']*100:.2f}%")
    print(f"  ROC-AUC   : {metrics['ROC_AUC']:.4f}")
    print(f"{dash}")
    print(f"  Confusion Matrix:")
    print(f"  [[{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"   [{cm[1,0]:4d}  {cm[1,1]:4d}]]")
    print(f"{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders + predict
# ══════════════════════════════════════════════════════════════════════════════

def predict_xgboost(models_dir: Path, X_test: np.ndarray) -> tuple:
    """Load XGBoost and return (y_pred, y_proba)."""
    path = models_dir / "xgboost.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run models/baselines.py first.")
    clf = joblib.load(path)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def predict_qsvm(models_dir: Path, X_test: np.ndarray, skip: bool = False) -> tuple:
    """Load QSVM artefacts, recompute test kernel, return (y_pred, y_proba)."""
    if skip:
        return None, None

    from models.baselines import build_quantum_kernel, compute_kernel_matrix

    for fname in ["qsvm.joblib", "qsvm_pca.joblib",
                  "qsvm_kernel_X_train.npy", "qsvm_max_abs.npy"]:
        if not (models_dir / fname).exists():
            raise FileNotFoundError(
                f"{models_dir / fname} not found. Run models/baselines.py first."
            )

    svc = joblib.load(models_dir / "qsvm.joblib")
    pca = joblib.load(models_dir / "qsvm_pca.joblib")
    kernel_X_train = np.load(models_dir / "qsvm_kernel_X_train.npy")
    max_abs = np.load(models_dir / "qsvm_max_abs.npy")

    X_test_pca = pca.transform(X_test)
    X_test_pca = X_test_pca / max_abs * np.pi

    print("  Recomputing QSVM test kernel matrix (needed for inference)...")
    kernel_fn = build_quantum_kernel()
    K_test = compute_kernel_matrix(kernel_fn, X_test_pca, kernel_X_train, verbose=True)

    y_pred = svc.predict(K_test)
    y_proba = svc.predict_proba(K_test)[:, 1]
    return y_pred, y_proba


def predict_tab_transformer(models_dir: Path, X_test: np.ndarray) -> tuple:
    """Load classical TabTransformer and return (y_pred, y_proba)."""
    path = models_dir / "tab_transformer.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run models/tab_transformer.py first.")

    # Import after checking
    sys.path.insert(0, str(BASE_DIR))
    from models.tab_transformer import TabTransformer

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = TabTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    X_t = torch.FloatTensor(X_test).to(DEVICE)
    with torch.no_grad():
        y_proba = torch.sigmoid(model(X_t)).squeeze(-1).cpu().numpy()
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


def predict_hybrid_transformer(models_dir: Path, X_test: np.ndarray) -> tuple:
    """Load Hybrid Quantum Transformer and return (y_pred, y_proba)."""
    path = models_dir / "hybrid_qt.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run models/hybrid_quantum_transformer.py first.")

    sys.path.insert(0, str(BASE_DIR))
    from models.hybrid_quantum_transformer import HybridTabTransformer

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = HybridTabTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    X_t = torch.FloatTensor(X_test).to(DEVICE)
    with torch.no_grad():
        y_proba = torch.sigmoid(model(X_t)).squeeze(-1).cpu().numpy()
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(
    cms: list, names: list, results_dir: Path
) -> None:
    """Save 1×4 confusion matrix grid at 300 dpi."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, cm, name, color in zip(axes, cms, names, COLORS):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not-CKD", "CKD"],
            yticklabels=["Not-CKD", "CKD"],
            ax=ax, linewidths=0.5, linecolor="gray",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

    fig.suptitle(
        "Confusion Matrices — CKD Test Set",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = results_dir / "confusion_matrices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  confusion_matrices.png saved (300 dpi)")


def plot_roc_curves(
    y_test: np.ndarray,
    all_proba: list,
    names: list,
    metrics_list: list,
    results_dir: Path,
) -> None:
    """Save overlaid ROC curves at 300 dpi."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for proba, name, color, metrics in zip(all_proba, names, COLORS, metrics_list):
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = metrics["ROC_AUC"]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — CKD Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    plt.tight_layout()
    out = results_dir / "roc_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  roc_curves.png saved (300 dpi)")


def plot_metrics_comparison(
    results_df: pd.DataFrame, results_dir: Path
) -> None:
    """Save grouped bar chart comparing all metrics at 300 dpi."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    n_models = len(results_df)
    n_metrics = len(metric_cols)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metric_cols]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=row["Model"], color=COLORS[i], alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — CKD Test Set",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=11)
    ax.set_ylim([0.5, 1.08])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = results_dir / "metrics_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  metrics_comparison.png saved (300 dpi)")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results_df: pd.DataFrame) -> None:
    """Print the comparison table with tabulate or pandas fallback."""
    print("\n" + "=" * 80)
    print("FULL COMPARISON TABLE")
    print("=" * 80)
    try:
        from tabulate import tabulate
        pct_cols = ["Accuracy", "Precision", "Recall", "F1"]
        display = results_df.copy()
        for col in pct_cols:
            display[col] = display[col].map(lambda x: f"{x*100:.2f}%")
        display["ROC_AUC"] = display["ROC_AUC"].map(lambda x: f"{x:.4f}")
        print(tabulate(display, headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        display = results_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            display[col] = display[col].map(lambda x: f"{x*100:.2f}%")
        display["ROC_AUC"] = display["ROC_AUC"].map(lambda x: f"{x:.4f}")
        print(display.to_string(index=False))
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(skip_quantum: bool = False) -> None:
    """Evaluate all models on the test set and produce all outputs."""
    print("=" * 60)
    print("STEP 5 — EVALUATION")
    print("=" * 60)
    print(f"Using device: {DEVICE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load test data ─────────────────────────────────────────────────────────
    for fname in ["X_test.npy", "y_test.npy"]:
        if not (DATA_DIR / fname).exists():
            raise FileNotFoundError(f"{DATA_DIR / fname} not found. Run preprocessing.py first.")

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    print(f"\nTest set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # ── Provenance verify (warning-only, backward compatible) ─────────────────
    if _HAS_Q1:
        prov_path = RESULTS_DIR / "provenance_log.json"
        if prov_path.exists():
            import json
            try:
                with open(prov_path) as f:
                    records = json.load(f)
                if isinstance(records, list):
                    for rec in records:
                        model_file = MODELS_DIR / rec.get("model_path", "")
                        if model_file.exists():
                            actual = compute_sha256(str(model_file))
                            if actual != rec.get("model_sha256", ""):
                                print(f"  [WARNING] SHA-256 mismatch: {model_file.name} "
                                      f"(checkpoint may have changed since signing)")
                        else:
                            pass  # model file not yet created — skip silently
            except Exception as e:
                print(f"  [WARNING] Provenance verification error: {e}")
        else:
            print("  [INFO] No provenance_log.json — run training to generate.")

    # ── Gather predictions ─────────────────────────────────────────────────────
    predictors = [
        ("XGBoost",                    predict_xgboost,          {}),
        ("Quantum SVM",                predict_qsvm,             {"skip": skip_quantum}),
        ("Classical TabTransformer",   predict_tab_transformer,  {}),
        ("Hybrid Quantum Transformer", predict_hybrid_transformer, {}),
    ]

    all_names, all_preds, all_proba, all_metrics, all_cms = [], [], [], [], []

    for name, predictor, kwargs in predictors:
        print(f"\nEvaluating: {name}")
        try:
            y_pred, y_proba = predictor(MODELS_DIR, X_test, **kwargs)
        except Exception as exc:
            print(f"  WARNING: Could not load {name}: {exc}")
            continue

        if y_pred is None:
            print(f"  Skipped ({name}).")
            continue

        metrics = compute_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        print_metrics_block(name, metrics, cm)

        all_names.append(name)
        all_preds.append(y_pred)
        all_proba.append(y_proba)
        all_metrics.append(metrics)
        all_cms.append(cm)

    if not all_names:
        print("ERROR: No models could be evaluated.")
        return

    # ── Comparison table ───────────────────────────────────────────────────────
    results_rows = []
    for name, metrics in zip(all_names, all_metrics):
        results_rows.append({
            "Model": name,
            "Accuracy":  round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall":    round(metrics["Recall"], 4),
            "F1":        round(metrics["F1"], 4),
            "ROC_AUC":   round(metrics["ROC_AUC"], 4),
        })

    results_df = pd.DataFrame(results_rows)
    print_comparison_table(results_df)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "results_table.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  results/results_table.csv saved")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_confusion_matrices(all_cms, all_names, RESULTS_DIR)
    plot_roc_curves(y_test, all_proba, all_names, all_metrics, RESULTS_DIR)
    plot_metrics_comparison(results_df, RESULTS_DIR)

    print(f"\n  confusion_matrices.png saved (300 dpi)")
    print(f"  roc_curves.png saved (300 dpi)")
    print(f"  metrics_comparison.png saved (300 dpi)")
    print(f"  results/results_table.csv saved")

    # ── Calibration analysis ───────────────────────────────────────────────────
    if _HAS_Q1 and len(all_proba) > 0:
        print("\nRunning calibration analysis...")
        try:
            models_probas = dict(zip(all_names, all_proba))
            cal_metrics = compute_calibration_metrics(y_test, models_probas)
            save_calibration_metrics(cal_metrics, str(RESULTS_DIR / "calibration_metrics.json"))
            reliability_diagram(y_test, models_probas,
                                out_path=str(RESULTS_DIR / "calibration_analysis.png"),
                                dataset_name="CKD")
            print(f"  results/calibration_metrics.json saved")
            print(f"  results/calibration_analysis.png saved")
        except Exception as exc:
            print(f"  [WARNING] Calibration analysis failed: {exc}")

    print("\nEvaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    skip_q = "--skip-quantum" in sys.argv
    run_evaluation(skip_quantum=skip_q)
