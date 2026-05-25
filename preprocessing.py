"""
preprocessing.py — Data loading, cleaning, imputation, encoding, SMOTE, and split.
UCI Chronic Kidney Disease (CKD) dataset — Step 1 of the QIP 2027 pipeline.
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "kidney_disease.csv"

# ── Feature definitions ────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
]
CAT_COLS = [
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane",
]
TARGET_COL = "classification"
ALL_FEATURE_COLS = NUMERIC_COLS + CAT_COLS  # 24 features


def ensure_dataset(csv_path: Path, data_dir: Path) -> None:
    """Download the CKD dataset from UCI ML Repository if not already present."""
    if csv_path.exists():
        print(f"Found existing dataset at: {csv_path}")
        return

    print("kidney_disease.csv not found — downloading from UCI ML Repository...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=336)
        df_feat = dataset.data.features
        df_tgt = dataset.data.targets
        df = pd.concat([df_feat, df_tgt], axis=1)
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Auto-download failed: {exc}\n"
            "Please manually download kidney_disease.csv from:\n"
            "  https://www.kaggle.com/datasets/mansoordaku/ckdisease\n"
            f"and place it at: {csv_path}"
        ) from exc


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load CSV, strip whitespace, replace '?', coerce numerics, drop id."""
    df = pd.read_csv(csv_path)

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalize ucimlrepo column names to canonical names
    df.rename(columns={
        "wbcc": "wc",   # white blood cell count
        "rbcc": "rc",   # red blood cell count
        "class": "classification",
    }, inplace=True)

    # Replace '?' sentinel with NaN
    df.replace("?", np.nan, inplace=True)

    # Strip string values in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan)

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def impute_and_encode(df: pd.DataFrame) -> tuple:
    """Impute missing values and ordinal-encode categoricals. Returns (X, y)."""
    # Separate present columns (dataset from ucimlrepo may differ slightly)
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    # Numeric imputation
    num_imputer = SimpleImputer(strategy="median")
    X_num = num_imputer.fit_transform(df[num_cols].values)

    # Categorical imputation
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_cat_raw = cat_imputer.fit_transform(df[cat_cols].values)

    # Ordinal encoding for categoricals
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    X_cat = enc.fit_transform(X_cat_raw)

    X = np.hstack([X_num, X_cat]).astype(np.float32)

    # Encode target
    target_raw = df[TARGET_COL].astype(str).str.strip().str.lower()
    target_map = {"ckd": 1, "notckd": 0, "ckd\t": 1}
    y = target_raw.map(target_map).values

    # Handle any unmapped values
    if np.any(pd.isna(y)):
        unmapped = target_raw[pd.isna(y)].unique()
        print(f"  Warning: unmapped target values: {unmapped}")
        # Try partial match
        y = np.array([
            1 if "ckd" in str(v) and "not" not in str(v) else (0 if "not" in str(v) else np.nan)
            for v in target_raw
        ])

    y = y.astype(np.int64)
    return X, y, num_cols, cat_cols


def run_preprocessing() -> None:
    """Full preprocessing pipeline — run this file directly."""
    print("=" * 60)
    print("STEP 1 — PREPROCESSING (SMOTE-leakage-free)")
    print("=" * 60)

    # ── Ensure dataset exists ──────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dataset(CSV_PATH, DATA_DIR)

    # ── Load and clean ─────────────────────────────────────────────────────────
    df = load_and_clean(CSV_PATH)
    print(f"\nShape of raw dataframe: {df.shape}")

    # ── Missing values ─────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing": missing.values,
        "Pct (%)": missing_pct.values,
    })
    missing_df = missing_df[missing_df["Missing"] > 0].reset_index(drop=True)
    print("\nMissing values per column:")
    print(missing_df.to_string(index=False))
    print(f"\nTotal columns with missing: {len(missing_df)}")

    # ── Impute and encode ──────────────────────────────────────────────────────
    X, y, num_cols, cat_cols = impute_and_encode(df)
    n_num = len(num_cols)

    print(f"\nAll feature columns ({len(num_cols + cat_cols)}):")
    for i, col in enumerate(num_cols + cat_cols, 1):
        print(f"  {i:2d}. {col}")

    # ── Save full arrays for cross-validation (pre-scale, pre-SMOTE) ──────────
    np.save(DATA_DIR / "X_full.npy", X)
    np.save(DATA_DIR / "y_full.npy", y)
    print(f"\nSaved X_full.npy {X.shape} and y_full.npy for 10-fold CV use.")

    # ── Class distribution (original) ─────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution (original dataset):")
    for cls, cnt in zip(unique, counts):
        label = "CKD (1)" if cls == 1 else "Not-CKD (0)"
        print(f"  {label}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # ── SPLIT FIRST on raw data — 70/15/15 stratified ─────────────────────────
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    ckd_tr = int(y_train_raw.sum()); notckd_tr = len(y_train_raw) - ckd_tr
    ckd_va = int(y_val.sum());       notckd_va = len(y_val) - ckd_va
    ckd_te = int(y_test.sum());      notckd_te = len(y_test) - ckd_te
    print(f"\n=== SPLIT SIZES (BEFORE SMOTE) ===")
    print(f"Train : X={X_train_raw.shape}  CKD={ckd_tr}  notCKD={notckd_tr}")
    print(f"Val   : X={X_val.shape}  CKD={ckd_va}  notCKD={notckd_va}")
    print(f"Test  : X={X_test.shape}  CKD={ckd_te}  notCKD={notckd_te}")

    # ── Scale numeric features (fit on raw train, transform val/test) ─────────
    scaler = StandardScaler()
    X_train_scaled = X_train_raw.copy()
    X_train_scaled[:, :n_num] = scaler.fit_transform(X_train_raw[:, :n_num])
    X_val[:, :n_num] = scaler.transform(X_val[:, :n_num])
    X_test[:, :n_num] = scaler.transform(X_test[:, :n_num])

    # ── Apply SMOTE ONLY to training set — val/test stay original ────────────
    print("\nApplying SMOTE to training set only (val/test untouched)...")
    smote = SMOTE(random_state=SEED)
    X_train, y_train = smote.fit_resample(X_train_scaled, y_train_raw)

    ckd_sm = int(y_train.sum()); notckd_sm = len(y_train) - ckd_sm
    print(f"\n=== AFTER SMOTE (train only) ===")
    print(f"Train : X={X_train.shape}  CKD={ckd_sm}  notCKD={notckd_sm}  (now balanced)")
    print(f"Val   : X={X_val.shape}  UNCHANGED")
    print(f"Test  : X={X_test.shape}  UNCHANGED")
    print(f"\nSMOTE leakage fix confirmed: val and test sets contain ONLY original samples.")

    # ── Save splits ────────────────────────────────────────────────────────────
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    for name, arr in splits.items():
        np.save(DATA_DIR / f"{name}.npy", arr)

    # Aliases for verification script
    np.save(DATA_DIR / "X_train_resampled.npy", X_train)
    np.save(DATA_DIR / "y_train_resampled.npy", y_train)

    joblib.dump(scaler, DATA_DIR / "scaler.joblib")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\nFinal split shapes:")
    print(f"  X_train (SMOTE'd) : {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val   (original): {X_val.shape}   y_val:   {y_val.shape}")
    print(f"  X_test  (original): {X_test.shape}  y_test:  {y_test.shape}")
    print(f"\nNumber of features: {X_train.shape[1]}")
    print("\nPreprocessing complete. Files saved.")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
