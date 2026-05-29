"""
tests/test_statistics.py -- Unit tests for utils/statistics.py.

Tests: DeLong AUC CI on known synthetic data, bootstrap CI width,
Wilcoxon test direction, Cohen's d, MCC, specificity, NPV.
"""

import numpy as np
import pytest


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_width_decreases_with_more_data():
    from utils.statistics import bootstrap_ci
    rng = np.random.default_rng(0)
    small = rng.standard_normal(20)
    large = rng.standard_normal(200)
    lo_s, hi_s = bootstrap_ci(small, n_boot=2000)
    lo_l, hi_l = bootstrap_ci(large, n_boot=2000)
    assert (hi_s - lo_s) > (hi_l - lo_l), "Larger sample should give narrower CI"


def test_bootstrap_ci_contains_true_mean():
    from utils.statistics import bootstrap_ci
    rng = np.random.default_rng(42)
    # True mean = 5.0; with 1000 samples the CI should reliably contain 5.0
    vals = rng.standard_normal(1000) + 5.0
    lo, hi = bootstrap_ci(vals, n_boot=5000)
    assert lo < 5.0 < hi, f"95% CI ({lo:.4f}, {hi:.4f}) should contain true mean=5.0"


def test_bootstrap_ci_tuple_length():
    from utils.statistics import bootstrap_ci
    lo, hi = bootstrap_ci([0.8, 0.9, 0.85, 0.88, 0.82])
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo <= hi


# ── DeLong AUC test ──────────────────────────────────────────────────────────

def test_delong_better_model_lower_p():
    from utils.statistics import delong_auc_test
    rng = np.random.default_rng(42)
    y_true = np.array([0]*100 + [1]*100)
    # Perfect model A
    proba_a = np.concatenate([rng.beta(2, 8, 100), rng.beta(8, 2, 100)])
    # Near-random model B
    proba_b = rng.uniform(0, 1, 200)

    result = delong_auc_test(y_true, proba_a, proba_b)
    assert "p_value" in result
    assert result["p_value"] < 0.05, "DeLong should detect significant AUC difference"


def test_delong_identical_models():
    from utils.statistics import delong_auc_test
    rng = np.random.default_rng(1)
    y_true = np.array([0]*50 + [1]*50)
    proba = rng.uniform(0, 1, 100)
    result = delong_auc_test(y_true, proba, proba)
    # Identical probas → p should be 1.0 (or very high)
    assert result["p_value"] > 0.05


# ── Wilcoxon signed-rank ─────────────────────────────────────────────────────

def test_wilcoxon_returns_expected_keys():
    from utils.statistics import wilcoxon_signed_rank
    a = [0.80, 0.85, 0.82, 0.88, 0.84, 0.79, 0.87, 0.83, 0.86, 0.81]
    b = [0.75, 0.78, 0.76, 0.80, 0.77, 0.74, 0.79, 0.76, 0.78, 0.75]
    result = wilcoxon_signed_rank(a, b)
    for key in ["statistic", "p_value", "cohen_d"]:
        assert key in result


def test_wilcoxon_detects_consistent_improvement():
    from utils.statistics import wilcoxon_signed_rank
    a = [0.90, 0.91, 0.92, 0.89, 0.93, 0.90, 0.91, 0.92, 0.90, 0.91]
    b = [0.80, 0.81, 0.79, 0.82, 0.80, 0.81, 0.80, 0.79, 0.82, 0.80]
    result = wilcoxon_signed_rank(a, b)
    assert result["p_value"] < 0.05, "Clear improvement should yield p < 0.05"


# ── Cohen's d ────────────────────────────────────────────────────────────────

def test_cohen_d_zero_for_equal_means():
    from utils.statistics import cohen_d
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    assert abs(cohen_d(a, b)) < 1e-10


def test_cohen_d_positive_direction():
    from utils.statistics import cohen_d
    a = [5.0, 6.0, 7.0]
    b = [1.0, 2.0, 3.0]
    assert cohen_d(a, b) > 0


# ── Classification metrics ────────────────────────────────────────────────────

def test_mcc_perfect():
    from utils.statistics import mcc
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    assert abs(mcc(y_true, y_pred) - 1.0) < 1e-10


def test_mcc_worst():
    from utils.statistics import mcc
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert mcc(y_true, y_pred) == pytest.approx(-1.0)


def test_specificity_all_negatives_correct():
    from utils.statistics import specificity
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    assert specificity(y_true, y_pred) == pytest.approx(1.0)


def test_npv_all_negatives_correct():
    from utils.statistics import npv
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert npv(y_true, y_pred) == pytest.approx(1.0)


def test_compute_full_metrics_keys():
    from utils.statistics import compute_full_metrics
    rng = np.random.default_rng(5)
    y_true = np.array([0]*50 + [1]*50)
    y_pred = rng.integers(0, 2, 100)
    y_prob = rng.uniform(0, 1, 100)
    result = compute_full_metrics(y_true, y_pred, y_prob)
    # Keys use full names in the actual implementation
    for key in ["accuracy", "f1", "auc_roc", "mcc", "kappa", "specificity", "npv", "brier", "auc_pr"]:
        assert key in result, f"Key '{key}' missing from compute_full_metrics output"
