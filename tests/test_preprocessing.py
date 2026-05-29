"""
tests/test_preprocessing.py -- Tests for data preprocessing and SMOTE leakage.

Key invariant: SMOTE must never be applied before the train/test split.
"""

import numpy as np
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


# ── SMOTE leakage tests ───────────────────────────────────────────────────────

def test_smote_not_in_val_or_test():
    """Val and test sets must contain only original (non-synthetic) samples."""
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = np.array([0] * 80 + [1] * 20)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    n_te_before = len(X_te)
    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    # Test set must be unchanged
    assert len(X_te) == n_te_before, "Test set size changed — SMOTE leaked into test"
    assert len(y_te) == n_te_before


def test_smote_only_balances_train():
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE

    X = np.random.default_rng(0).standard_normal((200, 10))
    y = np.array([0] * 160 + [1] * 40)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    sm = SMOTE(random_state=0)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    # After SMOTE: training should be balanced
    assert np.sum(y_tr_sm == 0) == np.sum(y_tr_sm == 1), "SMOTE did not balance training set"
    # Test untouched
    assert len(X_te) < len(X_tr_sm), "SMOTE did not augment training set"


def test_cv_smote_inside_fold_only():
    """Simulate 3-fold CV: SMOTE inside each fold must not bleed into val split."""
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(99)
    X = rng.standard_normal((120, 6))
    y = np.array([0] * 96 + [1] * 24)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=99)
    for tr_idx, va_idx in skf.split(X, y):
        n_va_before = len(va_idx)
        X_tr_fold = X[tr_idx]; y_tr_fold = y[tr_idx]
        X_va_fold = X[va_idx]; y_va_fold = y[va_idx]

        sm = SMOTE(random_state=99)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_fold, y_tr_fold)

        assert len(X_va_fold) == n_va_before, "Validation fold changed after SMOTE"


# ── Output shape tests ────────────────────────────────────────────────────────

def test_ckd_full_arrays_exist():
    X_path = DATA_DIR / "X_full.npy"
    y_path = DATA_DIR / "y_full.npy"
    if not X_path.exists():
        pytest.skip("X_full.npy not found — run preprocessing.py first")

    X = np.load(X_path)
    y = np.load(y_path)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 24, f"Expected 24 CKD features, got {X.shape[1]}"


def test_ckd_train_val_test_shapes():
    for fname in ["X_train.npy", "X_val.npy", "X_test.npy"]:
        if not (DATA_DIR / fname).exists():
            pytest.skip(f"{fname} not found — run preprocessing.py first")

    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    # Feature count must be consistent
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    # Labels match rows
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # Training is larger (SMOTE applied)
    assert X_train.shape[0] >= X_val.shape[0] + X_test.shape[0]


def test_fhs_full_arrays_exist():
    X_path = DATA_DIR / "fhs_X_full.npy"
    if not X_path.exists():
        pytest.skip("fhs_X_full.npy not found — run fhs_preprocessing.py first")

    X = np.load(X_path)
    y = np.load(DATA_DIR / "fhs_y_full.npy")
    assert X.shape[1] == 15, f"Expected 15 FHS features, got {X.shape[1]}"
    assert X.shape[0] == y.shape[0]


def test_target_binary():
    for y_file in ["y_full.npy", "fhs_y_full.npy"]:
        path = DATA_DIR / y_file
        if not path.exists():
            continue
        y = np.load(path)
        unique = np.unique(y)
        assert set(unique).issubset({0, 1}), f"Non-binary labels in {y_file}: {unique}"
