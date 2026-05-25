"""
fhs_train_models.py -- Train all models on FHS data.
Identical hyperparameters to CKD; only n_features changes (24->15).
Step FHS-2 of the QIP 2027 dual-dataset pipeline.
"""

import argparse
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(BASE_DIR))
from models.tab_transformer import TabTransformer, EarlyStopping as ES_TT
from models.hybrid_quantum_transformer import HybridTabTransformer, EarlyStopping as ES_HQ


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=gen if shuffle else None)


def _train_nn(model, X_train, y_train, X_val, y_val, lr, epochs, batch_size, patience, label):
    """Generic training loop for TabTransformer / HybridTabTransformer."""
    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    early_stop = ES_TT(patience=patience)

    best_val_loss = float("inf")
    best_state: dict = {}
    best_epoch = 1
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining {label} for up to {epochs} epochs (patience={patience}):")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb).squeeze(-1)
                val_losses.append(criterion(logits, yb).item())
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(yb.long().cpu().numpy())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(avg_val)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch:3d}/{epochs}] | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Acc: {val_acc*100:.2f}%")
        sys.stdout.flush()

        if early_stop(avg_val):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"\nBest Val Accuracy: {max(history['val_acc'])*100:.2f}% at epoch {best_epoch}")
    return model, history


def train_xgboost(data_dir: Path, models_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("FHS XGBoost")
    print("=" * 60)
    X_train = np.load(data_dir / "fhs_X_train.npy")
    y_train = np.load(data_dir / "fhs_y_train.npy")
    X_val = np.load(data_dir / "fhs_X_val.npy")
    y_val = np.load(data_dir / "fhs_y_val.npy")
    X_test = np.load(data_dir / "fhs_X_test.npy")
    y_test = np.load(data_dir / "fhs_y_test.npy")

    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        eval_metric="logloss", tree_method="hist", device="cpu",
    )
    clf.fit(X_train, y_train, verbose=False)

    for split_name, Xs, ys in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = clf.predict(Xs)
        proba = clf.predict_proba(Xs)[:, 1]
        acc = accuracy_score(ys, preds)
        f1 = f1_score(ys, preds)
        auc = roc_auc_score(ys, proba)
        print(f"  {split_name}: Acc={acc*100:.2f}%  F1={f1*100:.2f}%  AUC={auc:.4f}")

    joblib.dump(clf, models_dir / "fhs_xgboost.joblib")
    print(f"  Saved: models/fhs_xgboost.joblib")


def train_tab_transformer(data_dir: Path, models_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("FHS Classical TabTransformer")
    print("=" * 60)
    print(f"Using device: {DEVICE}")
    X_train = np.load(data_dir / "fhs_X_train.npy")
    y_train = np.load(data_dir / "fhs_y_train.npy")
    X_val = np.load(data_dir / "fhs_X_val.npy")
    y_val = np.load(data_dir / "fhs_y_val.npy")
    X_test = np.load(data_dir / "fhs_X_test.npy")
    y_test = np.load(data_dir / "fhs_y_test.npy")

    n_features = X_train.shape[1]
    config = {"n_features": n_features, "d_model": 32, "n_heads": 4,
              "n_layers": 2, "dim_ff": 128, "dropout": 0.1}
    model = TabTransformer(**config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_features={n_features}  Trainable params: {total_params:,}")

    model, history = _train_nn(
        model, X_train, y_train, X_val, y_val,
        lr=1e-3, epochs=50, batch_size=32, patience=10,
        label="FHS TabTransformer"
    )

    model.eval()
    with torch.no_grad():
        Xt = torch.FloatTensor(X_test).to(DEVICE)
        logits = model(Xt).squeeze(-1)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        proba = torch.sigmoid(logits).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"  Test: Acc={acc*100:.2f}%  F1={f1*100:.2f}%  AUC={auc:.4f}")

    torch.save({"model_state": model.state_dict(), "config": config, "history": history},
               models_dir / "fhs_tab_transformer.pt")
    print(f"  Saved: models/fhs_tab_transformer.pt")


def train_hybrid_transformer(data_dir: Path, models_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("FHS Hybrid Quantum Transformer")
    print("=" * 60)
    print(f"Using device: {DEVICE}")
    X_train = np.load(data_dir / "fhs_X_train.npy")
    y_train = np.load(data_dir / "fhs_y_train.npy")
    X_val = np.load(data_dir / "fhs_X_val.npy")
    y_val = np.load(data_dir / "fhs_y_val.npy")
    X_test = np.load(data_dir / "fhs_X_test.npy")
    y_test = np.load(data_dir / "fhs_y_test.npy")

    n_features = X_train.shape[1]
    config = {"n_features": n_features, "d_model": 32, "n_heads": 4,
              "n_layers": 2, "dropout": 0.1}
    model = HybridTabTransformer(**config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_features={n_features}  Trainable params: {total_params:,}")

    model, history = _train_nn(
        model, X_train, y_train, X_val, y_val,
        lr=5e-4, epochs=50, batch_size=32, patience=10,
        label="FHS HybridQT"
    )

    model.eval()
    with torch.no_grad():
        Xt = torch.FloatTensor(X_test).to(DEVICE)
        logits = model(Xt).squeeze(-1)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        proba = torch.sigmoid(logits).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"  Test: Acc={acc*100:.2f}%  F1={f1*100:.2f}%  AUC={auc:.4f}")

    torch.save({"model_state": model.state_dict(), "config": config, "history": history},
               models_dir / "fhs_hybrid_qt.pt")
    print(f"  Saved: models/fhs_hybrid_qt.pt")


def train_qsvm(data_dir: Path, models_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("FHS QSVM")
    print("=" * 60)
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from models.baselines import build_quantum_kernel, compute_kernel_matrix

    X_train = np.load(data_dir / "fhs_X_train.npy")
    y_train = np.load(data_dir / "fhs_y_train.npy")
    X_val = np.load(data_dir / "fhs_X_val.npy")
    y_val = np.load(data_dir / "fhs_y_val.npy")
    X_test = np.load(data_dir / "fhs_X_test.npy")
    y_test = np.load(data_dir / "fhs_y_test.npy")

    qkernel = build_quantum_kernel()

    # Stratified 50/class subset from training
    rng = np.random.RandomState(SEED)
    idx0 = np.where(y_train == 0)[0]; idx1 = np.where(y_train == 1)[0]
    sub_size = min(50, len(idx0), len(idx1))
    sub_idx = np.concatenate([
        rng.choice(idx0, sub_size, replace=False),
        rng.choice(idx1, sub_size, replace=False),
    ])
    X_sub = X_train[sub_idx]; y_sub = y_train[sub_idx]

    # PCA(4)
    pca = PCA(n_components=4, random_state=SEED)
    X_sub_pca = pca.fit_transform(X_sub)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # Normalize to [-pi, pi]
    norm_max = np.abs(X_sub_pca).max(axis=0) + 1e-8
    X_sub_n = X_sub_pca / norm_max * np.pi
    X_val_n = X_val_pca / norm_max * np.pi
    X_test_n = X_test_pca / norm_max * np.pi

    print(f"  Computing kernel matrices (subset={len(X_sub)}, val={len(X_val)}, test={len(X_test)})...")
    K_train = compute_kernel_matrix(X_sub_n, X_sub_n, qkernel)
    K_val = compute_kernel_matrix(X_val_n, X_sub_n, qkernel)
    K_test = compute_kernel_matrix(X_test_n, X_sub_n, qkernel)

    svm = SVC(kernel="precomputed", probability=True, C=10.0, random_state=SEED)
    svm.fit(K_train, y_sub)

    for split_name, K, ys in [("Val", K_val, y_val), ("Test", K_test, y_test)]:
        preds = svm.predict(K)
        proba = svm.predict_proba(K)[:, 1]
        acc = accuracy_score(ys, preds)
        f1 = f1_score(ys, preds)
        auc = roc_auc_score(ys, proba)
        print(f"  {split_name}: Acc={acc*100:.2f}%  F1={f1*100:.2f}%  AUC={auc:.4f}")

    joblib.dump({"svm": svm, "pca": pca, "norm_max": norm_max, "qkernel": qkernel,
                 "X_sub_n": X_sub_n, "y_sub": y_sub},
                models_dir / "fhs_qsvm.joblib")
    print(f"  Saved: models/fhs_qsvm.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models on FHS data")
    parser.add_argument("--skip-qsvm", action="store_true", help="Skip QSVM training")
    args = parser.parse_args()

    print("=" * 60)
    print("FHS STEP 2 -- MODEL TRAINING")
    print("=" * 60)

    for fname in ["fhs_X_train.npy", "fhs_X_val.npy", "fhs_X_test.npy",
                  "fhs_y_train.npy", "fhs_y_val.npy", "fhs_y_test.npy"]:
        if not (DATA_DIR / fname).exists():
            raise FileNotFoundError(f"{DATA_DIR / fname} not found. Run fhs_preprocessing.py first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_xgboost(DATA_DIR, MODELS_DIR)
    train_tab_transformer(DATA_DIR, MODELS_DIR)
    train_hybrid_transformer(DATA_DIR, MODELS_DIR)

    if not args.skip_qsvm:
        train_qsvm(DATA_DIR, MODELS_DIR)
    else:
        print("\nQSVM skipped (--skip-qsvm).")

    print("\n" + "=" * 60)
    print("FHS model training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
