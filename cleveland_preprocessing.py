"""
cleveland_preprocessing.py -- Cleveland Heart Disease dataset preprocessing.

n=303 (~297 after dropping '?' rows), 13 features, binary target (disease present
if num > 0). Well-cited medical ML benchmark. Mirrors preprocessing.py:
median imputation, StandardScaler, SMOTE in train folds only.

Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
         slope, ca, thal, num (target)

Q1 upgrade: SHA-256 integrity hash logged to results/data_hashes.json.
"""

import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from utils.integrity import log_data_hash
    _HAS_INTEGRITY = True
except ImportError:
    _HAS_INTEGRITY = False

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "cleveland_heart.csv"
CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# ── Feature definitions ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COL = "num"
N_NUM = len(FEATURE_COLS)  # 13 — all treated as numeric (small integer codes + continuous)


def ensure_dataset(csv_path: Path, data_dir: Path) -> None:
    """Download the Cleveland dataset from UCI ML Repository if not present."""
    if csv_path.exists():
        print(f"Found existing dataset at: {csv_path}")
        return

    print("cleveland_heart.csv not found — downloading from UCI ML Repository...")
    cols = FEATURE_COLS + [TARGET_COL]
    try:
        df = pd.read_csv(CLEVELAND_URL, header=None, names=cols)
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Auto-download failed: {exc}\n"
            "Please manually download processed.cleveland.data from:\n"
            "  https://archive.ics.uci.edu/ml/datasets/Heart+Disease\n"
            f"and place it at: {csv_path} "
            f"(columns: {', '.join(cols)})"
        ) from exc


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load CSV, replace '?' with NaN, drop those rows, coerce numeric."""
    if _HAS_INTEGRITY:
        registry_path = BASE_DIR / "results" / "data_hashes.json"
        digest = log_data_hash(str(csv_path), str(registry_path), label="cleveland_raw")
        print(f"  [integrity] Cleveland SHA-256: {digest[:16]}... (full hash in data_hashes.json)")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # '?' sentinel -> NaN
    df.replace("?", np.nan, inplace=True)

    # Coerce all columns numeric
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values (the ~6 rows that had '?')
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows containing '?' / missing values.")

    return df


def impute_and_encode(df: pd.DataFrame) -> tuple:
    """Median-impute (defensive), binarize target. Returns (X, y)."""
    num_imputer = SimpleImputer(strategy="median")
    X = num_imputer.fit_transform(df[FEATURE_COLS].values).astype(np.float32)
    # Binarize: disease present if severity num > 0
    y = (df[TARGET_COL].values > 0).astype(np.int64)
    return X, y


def run_preprocessing() -> None:
    """Full Cleveland preprocessing pipeline."""
    print("=" * 60)
    print("CLEVELAND HEART DISEASE PREPROCESSING (SMOTE-leakage-free)")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dataset(CSV_PATH, DATA_DIR)

    df = load_and_clean(CSV_PATH)
    print(f"\nShape of cleaned dataframe: {df.shape}")

    X, y = impute_and_encode(df)
    print(f"\nFeature columns ({N_NUM}):")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {col}")

    # Save full arrays for cross-validation (pre-scale, pre-SMOTE)
    np.save(DATA_DIR / "cleveland_X_full.npy", X)
    np.save(DATA_DIR / "cleveland_y_full.npy", y)
    print(f"\nSaved cleveland_X_full.npy {X.shape} and cleveland_y_full.npy for 10-fold CV use.")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution (original, binarized):")
    for cls, cnt in zip(unique, counts):
        label = "Disease (1)" if cls == 1 else "No-Disease (0)"
        print(f"  {label}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # Split FIRST on raw data — 70/15/15 stratified
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"\n=== SPLIT SIZES (BEFORE SMOTE) ===")
    print(f"Train : X={X_train_raw.shape}  pos={int(y_train_raw.sum())}")
    print(f"Val   : X={X_val.shape}  pos={int(y_val.sum())}")
    print(f"Test  : X={X_test.shape}  pos={int(y_test.sum())}")

    # Scale (all features numeric); fit on raw train, transform val/test
    scaler = StandardScaler()
    X_train_scaled = X_train_raw.copy()
    X_train_scaled[:, :N_NUM] = scaler.fit_transform(X_train_raw[:, :N_NUM])
    X_val[:, :N_NUM] = scaler.transform(X_val[:, :N_NUM])
    X_test[:, :N_NUM] = scaler.transform(X_test[:, :N_NUM])

    # SMOTE only on training set
    print("\nApplying SMOTE to training set only (val/test untouched)...")
    smote = SMOTE(random_state=SEED)
    X_train, y_train = smote.fit_resample(X_train_scaled, y_train_raw)
    print(f"Train after SMOTE: X={X_train.shape}  pos={int(y_train.sum())}  (balanced)")

    splits = {
        "cleveland_X_train": X_train, "cleveland_X_val": X_val, "cleveland_X_test": X_test,
        "cleveland_y_train": y_train, "cleveland_y_val": y_val, "cleveland_y_test": y_test,
    }
    for name, arr in splits.items():
        np.save(DATA_DIR / f"{name}.npy", arr)
    joblib.dump(scaler, DATA_DIR / "cleveland_scaler.joblib")

    print("\nFinal split shapes:")
    print(f"  cleveland_X_train (SMOTE'd) : {X_train.shape}")
    print(f"  cleveland_X_val   (original): {X_val.shape}")
    print(f"  cleveland_X_test  (original): {X_test.shape}")
    print(f"\nNumber of features: {X_train.shape[1]}")
    print("\nCleveland preprocessing complete. Files saved.")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
