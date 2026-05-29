"""
fhs_preprocessing.py -- Data loading, cleaning, imputation, scaling, SMOTE, and split.
Framingham Heart Study (FHS) dataset -- Step FHS-1 of the QIP 2027 dual-dataset pipeline.

Feature layout (15 features total):
  Indices 0-7:  continuous (age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose)
  Indices 8-14: binary/ordinal (male, currentSmoker, BPMeds, prevalentStroke, prevalentHyp,
                diabetes, education)
Target: TenYearCHD (binary, ~85/15 imbalance)
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "framingham.csv"

CONTINUOUS_COLS = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
BINARY_COLS = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "education"]
ALL_FEATURE_COLS = CONTINUOUS_COLS + BINARY_COLS  # 15 features total
TARGET_COL = "TenYearCHD"
N_NUM = len(CONTINUOUS_COLS)  # 8


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load CSV, strip whitespace, handle missing values."""
    if _HAS_INTEGRITY:
        registry_path = Path(__file__).parent / "results" / "data_hashes.json"
        digest = log_data_hash(str(csv_path), str(registry_path), label="fhs_raw")
        print(f"  [integrity] FHS SHA-256: {digest[:16]}... (full hash in data_hashes.json)")

    df = pd.read_csv(csv_path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df.columns = df.columns.str.strip()

    df.replace("?", np.nan, inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan)

    for col in CONTINUOUS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def impute_and_assemble(df: pd.DataFrame) -> tuple:
    """Impute missing values and assemble feature matrix. Returns (X, y)."""
    cont_cols = [c for c in CONTINUOUS_COLS if c in df.columns]
    bin_cols = [c for c in BINARY_COLS if c in df.columns]

    cont_imputer = SimpleImputer(strategy="median")
    X_cont = cont_imputer.fit_transform(df[cont_cols].values)

    bin_imputer = SimpleImputer(strategy="most_frequent")
    X_bin = bin_imputer.fit_transform(df[bin_cols].values)

    X = np.hstack([X_cont, X_bin]).astype(np.float32)

    y = df[TARGET_COL].values.astype(np.int64)
    return X, y, cont_cols, bin_cols


def run_preprocessing() -> None:
    """Full FHS preprocessing pipeline."""
    print("=" * 60)
    print("FHS STEP 1 -- PREPROCESSING (SMOTE-leakage-free)")
    print("=" * 60)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"\nframingham.csv not found at: {CSV_PATH}\n"
            "Please download from Kaggle:\n"
            "  https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset\n"
            f"and place it at: {CSV_PATH}"
        )

    print(f"\nLoading dataset from: {CSV_PATH}")
    df = load_and_clean(CSV_PATH)
    print(f"Shape of raw dataframe: {df.shape}")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing": missing.values,
        "Pct (%)": missing_pct.values,
    })
    missing_df = missing_df[missing_df["Missing"] > 0].reset_index(drop=True)
    if len(missing_df) > 0:
        print("\nMissing values per column:")
        print(missing_df.to_string(index=False))
    else:
        print("\nNo missing values found.")

    X, y, cont_cols, bin_cols = impute_and_assemble(df)
    feature_cols = cont_cols + bin_cols

    print(f"\nAll feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        kind = "continuous" if i <= len(cont_cols) else "binary/ordinal"
        print(f"  {i:2d}. {col}  [{kind}]")

    # Save pre-scale, pre-SMOTE arrays for 10-fold CV use
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "fhs_X_full.npy", X)
    np.save(DATA_DIR / "fhs_y_full.npy", y)
    print(f"\nSaved fhs_X_full.npy {X.shape} and fhs_y_full.npy for 10-fold CV use.")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution (original dataset):")
    for cls, cnt in zip(unique, counts):
        label = "CHD (1)" if cls == 1 else "No-CHD (0)"
        print(f"  {label}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # Split FIRST: 70/15/15 stratified
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    chd_tr = int(y_train_raw.sum()); nchd_tr = len(y_train_raw) - chd_tr
    chd_va = int(y_val.sum());       nchd_va = len(y_val) - chd_va
    chd_te = int(y_test.sum());      nchd_te = len(y_test) - chd_te
    print(f"\n=== SPLIT SIZES (BEFORE SMOTE) ===")
    print(f"Train : X={X_train_raw.shape}  CHD={chd_tr}  noCHD={nchd_tr}")
    print(f"Val   : X={X_val.shape}  CHD={chd_va}  noCHD={nchd_va}")
    print(f"Test  : X={X_test.shape}  CHD={chd_te}  noCHD={nchd_te}")

    # Scale: fit on raw train continuous cols only
    scaler = StandardScaler()
    X_train_scaled = X_train_raw.copy()
    X_train_scaled[:, :N_NUM] = scaler.fit_transform(X_train_raw[:, :N_NUM])
    X_val_s = X_val.copy()
    X_val_s[:, :N_NUM] = scaler.transform(X_val[:, :N_NUM])
    X_test_s = X_test.copy()
    X_test_s[:, :N_NUM] = scaler.transform(X_test[:, :N_NUM])

    # SMOTE on training set ONLY
    print("\nApplying SMOTE to training set only (val/test untouched)...")
    smote = SMOTE(sampling_strategy=1.0, random_state=SEED)
    X_train, y_train = smote.fit_resample(X_train_scaled, y_train_raw)

    chd_sm = int(y_train.sum()); nchd_sm = len(y_train) - chd_sm
    print(f"\n=== AFTER SMOTE (train only) ===")
    print(f"Train : X={X_train.shape}  CHD={chd_sm}  noCHD={nchd_sm}  (now balanced)")
    print(f"Val   : X={X_val_s.shape}  UNCHANGED")
    print(f"Test  : X={X_test_s.shape}  UNCHANGED")
    print(f"\nSMOTE leakage fix confirmed: val and test sets contain ONLY original samples.")

    # Save splits
    np.save(DATA_DIR / "fhs_X_train.npy", X_train)
    np.save(DATA_DIR / "fhs_y_train.npy", y_train)
    np.save(DATA_DIR / "fhs_X_val.npy", X_val_s)
    np.save(DATA_DIR / "fhs_y_val.npy", y_val)
    np.save(DATA_DIR / "fhs_X_test.npy", X_test_s)
    np.save(DATA_DIR / "fhs_y_test.npy", y_test)
    joblib.dump(scaler, DATA_DIR / "fhs_scaler.joblib")

    print("\nFinal split shapes:")
    print(f"  fhs_X_train (SMOTE'd) : {X_train.shape}  fhs_y_train: {y_train.shape}")
    print(f"  fhs_X_val   (original): {X_val_s.shape}  fhs_y_val:   {y_val.shape}")
    print(f"  fhs_X_test  (original): {X_test_s.shape}  fhs_y_test:  {y_test.shape}")
    print(f"\nNumber of features: {X_train.shape[1]}")
    print("\nFHS preprocessing complete. Files saved.")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
