"""
scripts/resign_checkpoints.py -- Re-sign model checkpoints with a provenance note.

Writes a FRESH results/provenance_log.json (truncating any prior log to avoid
duplicate appends), signing every models/*.pt and models/*.joblib with a
metadata note clarifying that these checkpoints predate the AdaptiveVQCSelector
refactor and that reported CV metrics come from the full pipeline, not from
loading a checkpoint.

Run: python scripts/resign_checkpoints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.integrity import sign_model, save_provenance_record  # noqa: E402

MODELS_DIR = BASE_DIR / "models"
PROV_PATH = BASE_DIR / "results" / "provenance_log.json"

NOTE = (
    "Pre-AdaptiveVQCSelector checkpoint. Reported cross-validation metrics are "
    "produced by the full pipeline (per-fold adaptive circuit), not by loading "
    "this checkpoint. See REPRODUCIBILITY.md (Model Checkpoint Notes)."
)

# Friendly metadata per known checkpoint (others fall back to a generic label)
META = {
    "hybrid_qt.pt":          {"model": "CKD HybridQT",        "dataset": "ckd"},
    "tab_transformer.pt":    {"model": "CKD TabTransformer",  "dataset": "ckd"},
    "xgboost.joblib":        {"model": "CKD XGBoost",         "dataset": "ckd"},
    "qsvm.joblib":           {"model": "CKD QSVM",            "dataset": "ckd"},
    "qsvm_pca.joblib":       {"model": "CKD QSVM PCA",        "dataset": "ckd"},
    "fhs_hybrid_qt.pt":      {"model": "FHS HybridQT",        "dataset": "fhs"},
    "fhs_tab_transformer.pt":{"model": "FHS TabTransformer",  "dataset": "fhs"},
    "fhs_xgboost.joblib":    {"model": "FHS XGBoost",         "dataset": "fhs"},
}


def main() -> None:
    PROV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Truncate any existing log so we don't accumulate duplicate records.
    if PROV_PATH.exists():
        PROV_PATH.unlink()

    ckpts = sorted(p for p in MODELS_DIR.iterdir()
                   if p.suffix in (".pt", ".joblib"))
    if not ckpts:
        print(f"No checkpoints found in {MODELS_DIR}")
        return

    print("=" * 60)
    print("RE-SIGNING MODEL CHECKPOINTS (with provenance note)")
    print("=" * 60)
    for p in ckpts:
        meta = dict(META.get(p.name, {"model": p.stem, "dataset": "unknown"}))
        meta["note"] = NOTE
        record = sign_model(str(p), meta)
        save_provenance_record(record, str(PROV_PATH))
        print(f"  signed {p.name:24s} {record['model_sha256'][:16]}...")

    print(f"\nWrote {len(ckpts)} records -> {PROV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
