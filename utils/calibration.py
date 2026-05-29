"""
utils/calibration.py -- Probability calibration analysis for Q1 medical AI papers.

Includes:
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Reliability diagrams (matplotlib)
  - Platt scaling and isotonic regression post-hoc calibration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression


# ── ECE / MCE ────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error: weighted mean |accuracy - confidence| per bin.
    """
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def max_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error: max |accuracy - confidence| across bins."""
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        diff = abs(y_true[mask].mean() - y_prob[mask].mean())
        mce = max(mce, diff)
    return float(mce)


# ── Calibration (Platt + isotonic) ───────────────────────────────────────────

def platt_scaling(
    y_cal: np.ndarray,
    proba_cal: np.ndarray,
    proba_test: np.ndarray,
) -> np.ndarray:
    """
    Fit a logistic regression on calibration probabilities and apply to test.
    Returns calibrated probabilities for the test set.
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(proba_cal.reshape(-1, 1), y_cal)
    return lr.predict_proba(proba_test.reshape(-1, 1))[:, 1]


def isotonic_calibration(
    y_cal: np.ndarray,
    proba_cal: np.ndarray,
    proba_test: np.ndarray,
) -> np.ndarray:
    """
    Isotonic regression calibration. Returns calibrated probabilities for test.
    """
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba_cal, y_cal)
    return ir.transform(proba_test)


# ── Reliability diagrams ──────────────────────────────────────────────────────

def reliability_diagram(
    models_probas: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str,
    n_bins: int = 10,
    dataset_name: str = "",
) -> None:
    """
    Plot reliability diagrams for multiple models on one axis.

    models_probas: {model_name: (y_true, y_prob)}
    """
    MODEL_COLORS = {
        "XGBoost": "#2196F3",
        "TabTransformer": "#FF9800",
        "HybridQT": "#00D9FF",
        "QSVM": "#9C27B0",
        "LightGBM": "#4CAF50",
        "MLP": "#795548",
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect calibration")

    for name, (y_true, y_prob) in models_probas.items():
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        except Exception:
            continue
        color = MODEL_COLORS.get(name, "#555555")
        ece = expected_calibration_error(y_true, y_prob, n_bins)
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=5,
                label=f"{name} (ECE={ece:.3f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    title = f"Calibration Reliability Diagram"
    if dataset_name:
        title += f" — {dataset_name}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, linestyle="--", alpha=0.4)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [calibration] Saved: {out_path}")


# ── Full calibration report ───────────────────────────────────────────────────

def compute_calibration_metrics(
    models_probas: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
) -> dict:
    """
    For each model compute ECE, MCE, and optionally Platt-scaled ECE.
    Returns a dict suitable for saving to calibration_metrics.json.
    """
    report = {}
    for name, (y_true, y_prob) in models_probas.items():
        y_true = np.asarray(y_true, int)
        y_prob = np.asarray(y_prob, float)
        ece_raw = expected_calibration_error(y_true, y_prob, n_bins)
        mce_raw = max_calibration_error(y_true, y_prob, n_bins)

        # Platt calibration (train on same data — illustrative)
        try:
            prob_cal = platt_scaling(y_true, y_prob, y_prob)
            ece_platt = expected_calibration_error(y_true, prob_cal, n_bins)
        except Exception:
            ece_platt = None

        try:
            prob_iso = isotonic_calibration(y_true, y_prob, y_prob)
            ece_iso = expected_calibration_error(y_true, prob_iso, n_bins)
        except Exception:
            ece_iso = None

        report[name] = {
            "ece_raw": ece_raw,
            "mce_raw": mce_raw,
            "ece_platt_scaled": ece_platt,
            "ece_isotonic": ece_iso,
        }
    return report


def save_calibration_metrics(metrics: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2, default=str))
    print(f"  [calibration] Metrics saved: {out_path}")
