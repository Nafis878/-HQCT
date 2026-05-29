"""
models/baselines.py -- XGBoost, LightGBM, MLP, and Quantum SVM baselines.

XGBoost: 5-fold stratified CV, then final fit on full training data.
LightGBM: same protocol as XGBoost (modern gradient boosting baseline).
MLP: 2-layer feed-forward with GELU activations (depth-matched to TabTransformer).
QSVM: PCA to 4 dims, quantum kernel matrix, SVC(kernel='precomputed').

Step 4 of the QIP 2027 pipeline.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from utils.integrity import sign_model, save_provenance_record
    _HAS_INTEGRITY = True
except ImportError:
    _HAS_INTEGRITY = False


def _sign_and_log(model_path: Path, metadata: dict) -> None:
    if not _HAS_INTEGRITY:
        return
    try:
        results_dir = BASE_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        record = sign_model(str(model_path), metadata)
        save_provenance_record(record, str(results_dir / "provenance_log.json"))
        print(f"  [integrity] Provenance signed: {record['model_sha256'][:16]}...")
    except Exception as exc:
        print(f"  [WARNING] Provenance signing failed: {exc}")

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ── QSVM constants ─────────────────────────────────────────────────────────────
N_QUBITS = 4
QSVM_MAX_PER_CLASS = 50   # 50 per class = 100 total training samples for kernel


def _check_data(data_dir: Path) -> None:
    """Raise if preprocessed data arrays are missing."""
    for fname in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"{data_dir / fname} not found. Run preprocessing.py first."
            )


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(data_dir: Path = DATA_DIR, models_dir: Path = MODELS_DIR) -> dict:
    """Train XGBoost with 5-fold CV, then fit on full training set."""
    _check_data(data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
    )

    # 5-fold cross-validation on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")

    print("\nXGBoost 5-fold cross-validation:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score*100:.2f}%")
    print(f"  Mean CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    # Final fit with early-stopping hold-out (10%)
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=SEED
    )
    clf.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {test_acc*100:.2f}%")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, models_dir / "xgboost.joblib")
    print(f"  XGBoost model saved to {models_dir / 'xgboost.joblib'}")
    _sign_and_log(models_dir / "xgboost.joblib", {"model": "CKD XGBoost", "dataset": "ckd"})

    return {"model": clf, "cv_scores": cv_scores, "test_acc": test_acc}


# ══════════════════════════════════════════════════════════════════════════════
# Quantum Kernel for QSVM
# ══════════════════════════════════════════════════════════════════════════════

def build_quantum_kernel():
    """
    Create a quantum kernel using RY-angle embedding with CNOT entanglement.

    K(x1,x2) = |<phi(x1)|phi(x2)>|^2  where |phi(x)> = CNOT_ring · RY(x)|0>

    RY (not RZ) is essential: RZ only changes the phase of |0⟩ leaving
    measurement probabilities constant, so the kernel would be trivially 1.
    RY rotates in the X-Z plane, producing genuine superposition.
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        """
        Feature map U(x): RY angle embedding + CNOT ring entanglement.
        Kernel = |<0| U†(x2) U(x1) |0>|^2 = probs[|00...0>].
        """
        # U(x1)
        qml.AngleEmbedding(x1, wires=range(N_QUBITS), rotation="Y")
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        # U†(x2) — adjoint reverses gate order and negates angles
        for i in range(N_QUBITS - 1, -1, -1):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS), rotation="Y")
        return qml.probs(wires=range(N_QUBITS))

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        """Scalar fidelity K(x1,x2) ∈ [0,1]."""
        return float(kernel_circuit(x1, x2)[0])

    return kernel


def compute_kernel_matrix(
    kernel_fn, X1: np.ndarray, X2: np.ndarray, verbose: bool = True
) -> np.ndarray:
    """Compute Gram matrix K[i,j] = kernel(X1[i], X2[j])."""
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    total = n1 * n2
    done = 0
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j])
            done += 1
        if verbose and (i + 1) % 10 == 0:
            pct = (i + 1) / n1 * 100
            print(f"    Kernel progress: {done}/{total} ({pct:.0f}%)", flush=True)
    return K


def train_qsvm(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    skip: bool = False,
) -> dict:
    """Train a Quantum SVM with quantum kernel on a stratified subset."""
    if skip:
        print("  QSVM skipped (--skip-quantum flag).")
        return {}

    _check_data(data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    # PCA: reduce to N_QUBITS components then scale to [-π, π]
    pca = PCA(n_components=N_QUBITS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Normalise into [-π, π] for angle embedding
    max_abs = np.abs(X_train_pca).max(axis=0, keepdims=True) + 1e-8
    X_train_pca = X_train_pca / max_abs * np.pi
    X_test_pca = X_test_pca / max_abs * np.pi

    # Stratified subset for kernel matrix
    rng = np.random.default_rng(SEED)
    subset_idx = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        n_take = min(QSVM_MAX_PER_CLASS, len(cls_idx))
        chosen = rng.choice(cls_idx, n_take, replace=False)
        subset_idx.extend(chosen.tolist())
    subset_idx = np.array(subset_idx)

    kernel_X_train = X_train_pca[subset_idx]
    kernel_y_train = y_train[subset_idx]

    n_train_sub = len(kernel_X_train)
    print(f"\n  Quantum SVM:")
    print(f"  Computing quantum kernel matrix ({n_train_sub}x{n_train_sub})...")
    print(f"  This may take several minutes on CPU.")

    kernel_fn = build_quantum_kernel()

    t0 = time.perf_counter()
    K_train = compute_kernel_matrix(kernel_fn, kernel_X_train, kernel_X_train)
    t_train = time.perf_counter() - t0
    print(f"  Training kernel matrix shape: {K_train.shape}")
    print(f"  Kernel matrix computed in {t_train:.1f} seconds")

    # Add small diagonal regularisation for numerical stability
    K_train += 1e-6 * np.eye(n_train_sub)

    svc = SVC(kernel="precomputed", probability=True, random_state=SEED, C=10.0)
    svc.fit(K_train, kernel_y_train)

    # Test kernel matrix
    print(f"  Computing test kernel matrix ({len(X_test_pca)}x{n_train_sub})...")
    t1 = time.perf_counter()
    K_test = compute_kernel_matrix(kernel_fn, X_test_pca, kernel_X_train)
    t_test = time.perf_counter() - t1
    print(f"  Test kernel computed in {t_test:.1f} seconds")

    y_pred = svc.predict(K_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {test_acc*100:.2f}%")

    # Save all components needed for inference in evaluate.py
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(svc, models_dir / "qsvm.joblib")
    joblib.dump(pca, models_dir / "qsvm_pca.joblib")
    np.save(models_dir / "qsvm_kernel_X_train.npy", kernel_X_train)
    np.save(models_dir / "qsvm_max_abs.npy", max_abs)
    print(f"  QSVM artifacts saved to {models_dir}")

    return {
        "model": svc, "pca": pca,
        "kernel_X_train": kernel_X_train, "max_abs": max_abs,
        "test_acc": test_acc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LightGBM
# ══════════════════════════════════════════════════════════════════════════════

def train_lightgbm(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    save_name: str = "lightgbm.joblib",
) -> dict:
    """Train LightGBM with 5-fold CV + final fit; parallel to XGBoost protocol."""
    if not _HAS_LGB:
        print("  LightGBM not installed (pip install lightgbm); skipping.")
        return {}

    _check_data(data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")

    print("\nLightGBM 5-fold cross-validation:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score*100:.2f}%")
    print(f"  Mean CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=SEED
    )
    clf.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(20, verbose=False)])

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {test_acc*100:.2f}%")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, models_dir / save_name)
    print(f"  LightGBM model saved: {models_dir / save_name}")
    _sign_and_log(models_dir / save_name, {"model": "CKD LightGBM", "dataset": "ckd"})

    return {"model": clf, "cv_scores": cv_scores, "test_acc": test_acc}


# ══════════════════════════════════════════════════════════════════════════════
# MLP Baseline
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    2-layer feed-forward MLP depth-matched to TabTransformer.
    Architecture: Linear(n_feat, 128) -> GELU -> Linear(128, 64) -> GELU -> Linear(64, 1)
    """

    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(-1)


def train_mlp(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 10,
    save_name: str = "mlp.pt",
) -> MLP:
    """Train a 2-layer MLP baseline."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fname in ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"]:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(f"{data_dir / fname} not found.")

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    def _make_loader(X, y, shuffle):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(X_train, y_train, True)
    val_loader = _make_loader(X_val, y_val, False)

    n_features = X_train.shape[1]
    model = MLP(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0

    print(f"\nMLP training ({n_features}->{128}->{64}->1, {epochs} epochs):")
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb).squeeze(-1)
                val_losses.append(criterion(logits, yb).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
                val_labels.extend(yb.long().cpu().numpy())

        avg_val = float(np.mean(val_losses))
        val_acc = accuracy_score(val_labels, val_preds)

        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Val Loss: {avg_val:.4f} | Val Acc: {val_acc*100:.2f}%")
        sys.stdout.flush()

    model.load_state_dict(best_state)
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {"n_features": n_features, "hidden1": 128, "hidden2": 64},
    }, models_dir / save_name)
    print(f"  MLP saved: {models_dir / save_name}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_training(skip_quantum: bool = False) -> None:
    """Train XGBoost and QSVM baselines."""
    print("=" * 60)
    print("STEP 4 — BASELINES (XGBoost + QSVM)")
    print("=" * 60)

    print("\n[1/2] XGBoost")
    train_xgboost(DATA_DIR, MODELS_DIR)

    print("\n[2/2] Quantum SVM (QSVM)")
    train_qsvm(DATA_DIR, MODELS_DIR, skip=skip_quantum)

    print("\n[3/4] LightGBM")
    train_lightgbm(DATA_DIR, MODELS_DIR)

    print("\n[4/4] MLP")
    train_mlp(DATA_DIR, MODELS_DIR)

    print("\nBaselines complete. Results saved.")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    skip_q = "--skip-quantum" in sys.argv
    run_training(skip_quantum=skip_q)
