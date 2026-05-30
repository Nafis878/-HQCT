"""
pima_preprocessing.py -- PIMA Indians Diabetes dataset preprocessing.

n=768, 8 numeric features, binary Outcome (diabetes yes/no), ~35/65 imbalance.
Mirrors preprocessing.py: median imputation (zeros treated as missing for
physiologically-impossible columns), StandardScaler, SMOTE in train folds only.

Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
         DiabetesPedigreeFunction, Age, Outcome

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
CSV_PATH = DATA_DIR / "pima_diabetes.csv"
PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

# ── Feature definitions ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COL = "Outcome"
# Columns where a value of 0 is physiologically impossible -> treat as missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
N_NUM = len(FEATURE_COLS)  # 8 — all features are numeric


def ensure_dataset(csv_path: Path, data_dir: Path) -> None:
    """Download the PIMA dataset from a public mirror if not already present."""
    if csv_path.exists():
        print(f"Found existing dataset at: {csv_path}")
        return

    print("pima_diabetes.csv not found — downloading from public mirror...")
    cols = FEATURE_COLS + [TARGET_COL]
    try:
        df = pd.read_csv(PIMA_URL, header=None, names=cols)
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Auto-download failed: {exc}\n"
            "Please manually download the PIMA Indians Diabetes CSV from:\n"
            "  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            f"and place it at: {csv_path} "
            f"(columns: {', '.join(cols)})"
        ) from exc


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load CSV, treat physiologically-impossible zeros as NaN."""
    if _HAS_INTEGRITY:
        registry_path = BASE_DIR / "results" / "data_hashes.json"
        digest = log_data_hash(str(csv_path), str(registry_path), label="pima_raw")
        print(f"  [integrity] PIMA SHA-256: {digest[:16]}... (full hash in data_hashes.json)")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Coerce numeric
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Zeros that are physiologically impossible -> NaN (imputed later)
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    return df


def impute_and_encode(df: pd.DataFrame) -> tuple:
    """Median-impute features. Returns (X, y)."""
    num_imputer = SimpleImputer(strategy="median")
    X = num_imputer.fit_transform(df[FEATURE_COLS].values).astype(np.float32)
    y = df[TARGET_COL].astype(np.int64).values
    return X, y


def run_preprocessing() -> None:
    """Full PIMA preprocessing pipeline."""
    print("=" * 60)
    print("PIMA PREPROCESSING (SMOTE-leakage-free)")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dataset(CSV_PATH, DATA_DIR)

    df = load_and_clean(CSV_PATH)
    print(f"\nShape of raw dataframe: {df.shape}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\nMissing values (after zero->NaN):")
        print(missing.to_string())

    X, y = impute_and_encode(df)
    print(f"\nFeature columns ({N_NUM}):")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {col}")

    # Save full arrays for cross-validation (pre-scale, pre-SMOTE)
    np.save(DATA_DIR / "pima_X_full.npy", X)
    np.save(DATA_DIR / "pima_y_full.npy", y)
    print(f"\nSaved pima_X_full.npy {X.shape} and pima_y_full.npy for 10-fold CV use.")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution (original dataset):")
    for cls, cnt in zip(unique, counts):
        label = "Diabetes (1)" if cls == 1 else "No-Diabetes (0)"
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
        "pima_X_train": X_train, "pima_X_val": X_val, "pima_X_test": X_test,
        "pima_y_train": y_train, "pima_y_val": y_val, "pima_y_test": y_test,
    }
    for name, arr in splits.items():
        np.save(DATA_DIR / f"{name}.npy", arr)
    joblib.dump(scaler, DATA_DIR / "pima_scaler.joblib")

    print("\nFinal split shapes:")
    print(f"  pima_X_train (SMOTE'd) : {X_train.shape}")
    print(f"  pima_X_val   (original): {X_val.shape}")
    print(f"  pima_X_test  (original): {X_test.shape}")
    print(f"\nNumber of features: {X_train.shape[1]}")
    print("\nPIMA preprocessing complete. Files saved.")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
