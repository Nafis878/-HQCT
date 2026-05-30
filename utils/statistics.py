"""
utils/statistics.py -- Advanced statistical testing suite for Q1 journal submission.

All functions are pure (no side effects); they accept numpy arrays and return
dicts so callers can persist results to JSON freely.

Tests included:
  - Wilcoxon signed-rank (paired, non-parametric)
  - Friedman + Nemenyi post-hoc
  - DeLong AUC comparison (1988 covariance formula)
  - Bootstrap BCa confidence intervals
  - Cohen's d effect size
  - Matthews Correlation Coefficient
  - Cohen's Kappa
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)


# ── Wilcoxon signed-rank ──────────────────────────────────────────────────────

def wilcoxon_signed_rank(
    fold_scores_a: Sequence[float],
    fold_scores_b: Sequence[float],
) -> dict:
    """
    Wilcoxon signed-rank test (two-tailed, exact when n<=25).
    Returns statistic, p-value, and Cohen's d effect size.
    """
    a = np.array(fold_scores_a, dtype=float)
    b = np.array(fold_scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("fold_scores_a and fold_scores_b must have equal length")
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(stat),
        "p_value": float(p),
        "cohen_d": float(cohen_d(a, b)),
        "significant_005": bool(p < 0.05),
        "significant_001": bool(p < 0.01),
    }


# ── Friedman + Nemenyi ────────────────────────────────────────────────────────

def friedman_nemenyi(all_fold_scores: Dict[str, Sequence[float]]) -> dict:
    """
    Friedman test across all models, then Nemenyi post-hoc pairwise comparisons.
    all_fold_scores: {model_name: [fold_acc_1, ..., fold_acc_k]}
    Returns Friedman p-value and Nemenyi p-value matrix as nested dict.
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        return {"error": "scikit_posthocs not installed; run: pip install scikit-posthocs"}

    import pandas as pd

    models = list(all_fold_scores.keys())
    data = np.array([list(all_fold_scores[m]) for m in models]).T  # shape: (folds, models)
    friedman_stat, friedman_p = stats.friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])

    # Nemenyi post-hoc (requires DataFrame)
    df = pd.DataFrame(data, columns=models)
    nemenyi_df = sp.posthoc_nemenyi_friedman(df)

    nemenyi_dict = {}
    for m1 in models:
        nemenyi_dict[m1] = {}
        for m2 in models:
            nemenyi_dict[m1][m2] = float(nemenyi_df.loc[m1, m2])

    return {
        "test": "friedman_nemenyi",
        "friedman_statistic": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "friedman_significant_005": bool(friedman_p < 0.05),
        "nemenyi_p_matrix": nemenyi_dict,
        "models": models,
    }


# ── DeLong AUC test ───────────────────────────────────────────────────────────

def _delong_covariance(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray):
    """
    Compute covariance matrix of two AUC estimators via the DeLong 1988 method.
    Returns (auc_a, auc_b, cov_matrix).
    """
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))

    pos_a = proba_a[y_true == 1]
    neg_a = proba_a[y_true == 0]
    pos_b = proba_b[y_true == 1]
    neg_b = proba_b[y_true == 0]

    def placement(pos, neg):
        # Placement values: proportion of negatives below each positive
        pv = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
        nv = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])
        return pv, nv

    pv_a, nv_a = placement(pos_a, neg_a)
    pv_b, nv_b = placement(pos_b, neg_b)

    auc_a = float(np.mean(pv_a))
    auc_b = float(np.mean(pv_b))

    s10 = np.cov(np.stack([pv_a, pv_b]))
    s01 = np.cov(np.stack([nv_a, nv_b]))

    cov = s10 / n1 + s01 / n0
    return auc_a, auc_b, cov


def delong_auc_test(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
) -> dict:
    """
    DeLong 1988 test for comparing two ROC AUCs on the same dataset.
    Returns z-statistic, two-tailed p-value, and 95% CI for the AUC difference.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)

    auc_a, auc_b, cov = _delong_covariance(y_true, proba_a, proba_b)
    diff = auc_a - auc_b
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-12)))
    z = diff / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_lo = diff - 1.96 * se
    ci_hi = diff + 1.96 * se

    return {
        "test": "delong_auc",
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": diff,
        "z_statistic": z,
        "p_value": p,
        "ci_95_lower": ci_lo,
        "ci_95_upper": ci_hi,
        "significant_005": bool(p < 0.05),
    }


# ── Bootstrap BCa confidence intervals ───────────────────────────────────────

def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    statistic=np.mean,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """
    BCa (bias-corrected and accelerated) bootstrap CI.
    Returns (lower, upper) for the given statistic.
    """
    rng = np.random.RandomState(rng_seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    observed = float(statistic(arr))

    boot_stats = np.array([
        statistic(rng.choice(arr, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # Bias correction z0
    z0 = stats.norm.ppf(np.mean(boot_stats < observed) + 1e-12)

    # Acceleration a (jackknife)
    jk = np.array([statistic(np.delete(arr, i)) for i in range(n)])
    jk_mean = np.mean(jk)
    num = np.sum((jk_mean - jk) ** 3)
    den = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5 + 1e-12)
    a = num / den

    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    def _adj(z_a):
        return stats.norm.cdf(z0 + (z0 + z_a) / (1 - a * (z0 + z_a) + 1e-12))

    lo_p = _adj(z_alpha_lo)
    hi_p = _adj(z_alpha_hi)

    lo = float(np.percentile(boot_stats, 100 * lo_p))
    hi = float(np.percentile(boot_stats, 100 * hi_p))
    return lo, hi


def bootstrap_ci_dict(
    values: Sequence[float],
    metric_name: str = "metric",
    n_boot: int = 10_000,
) -> dict:
    lo, hi = bootstrap_ci(values, n_boot=n_boot)
    return {
        "metric": metric_name,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_95_lower": lo,
        "ci_95_upper": hi,
    }


# ── Effect size ───────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / (pooled_sd + 1e-12))


# ── Additional classification metrics ────────────────────────────────────────

def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Negative Rate = TN / (TN + FP)."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return tn / (tn + fp + 1e-12)


def npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative Predictive Value = TN / (TN + FN)."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn / (tn + fn + 1e-12)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    inference_ms: float = 0.0,
) -> dict:
    """
    Compute the full Q1-grade metric set for a single fold.
    Returns a flat dict ready to append to a DataFrame row.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score,
    )
    return {
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "f1":             float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_weighted":    float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_roc":        float(roc_auc_score(y_true, y_prob)),
        "auc_pr":         float(average_precision_score(y_true, y_prob)),
        "precision":      float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity":    specificity(y_true, y_pred),
        "npv":            npv(y_true, y_pred),
        "mcc":            mcc(y_true, y_pred),
        "kappa":          cohens_kappa(y_true, y_pred),
        "brier":          float(brier_score_loss(y_true, y_prob)),
        "log_loss":       float(log_loss(y_true, y_prob)),
        "inference_ms":   float(inference_ms),
    }


# ── All-pairs statistical comparison ─────────────────────────────────────────

def run_all_pairwise_tests(
    fold_scores: Dict[str, List[float]],
    fold_probas: Optional[Dict[str, np.ndarray]] = None,
    y_true: Optional[np.ndarray] = None,
    metric_name: str = "roc_auc",
) -> dict:
    """
    Run Wilcoxon, Friedman/Nemenyi, and optionally DeLong across all model pairs.
    Returns a single dict suitable for saving to statistical_tests.json.

    `metric_name` labels the per-fold scores in `fold_scores`. The CV pipelines
    pass per-fold ROC-AUC values, so the default is "roc_auc" (not "accuracy").
    """
    models = list(fold_scores.keys())
    results: dict = {"models": models, "pairwise_wilcoxon": {}, "friedman_nemenyi": {}}

    # Wilcoxon all pairs
    for i, m1 in enumerate(models):
        results["pairwise_wilcoxon"][m1] = {}
        for m2 in models:
            if m1 == m2:
                results["pairwise_wilcoxon"][m1][m2] = None
                continue
            try:
                results["pairwise_wilcoxon"][m1][m2] = wilcoxon_signed_rank(
                    fold_scores[m1], fold_scores[m2]
                )
            except Exception as e:
                results["pairwise_wilcoxon"][m1][m2] = {"error": str(e)}

    # Friedman + Nemenyi
    results["friedman_nemenyi"] = friedman_nemenyi(fold_scores)

    # Bootstrap CIs per model
    results["bootstrap_ci"] = {
        m: bootstrap_ci_dict(fold_scores[m], metric_name=metric_name)
        for m in models
    }

    # DeLong AUC comparison (HybridQT vs XGBoost if probas provided)
    if fold_probas is not None and y_true is not None and "HybridQT" in fold_probas and "XGBoost" in fold_probas:
        try:
            results["delong_hqct_vs_xgb"] = delong_auc_test(
                y_true, fold_probas["HybridQT"], fold_probas["XGBoost"]
            )
        except Exception as e:
            results["delong_hqct_vs_xgb"] = {"error": str(e)}

    return results


def save_statistical_tests(results: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, default=str))
