# Reproducibility Guide

All experiments in this repository are fully reproducible with fixed seeds.

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml && conda activate hqct

# 2. Get datasets (auto-downloads UCI CKD; FHS requires manual download)
python preprocessing.py          # CKD
python fhs_preprocessing.py      # FHS

# 3. Run full pipeline (skip quantum for speed)
python main.py --skip-quantum
python main_fhs.py --skip-quantum

# 4. 10-fold CV evaluation
python cv_evaluation.py --skip-qsvm
python fhs_cv_evaluation.py --skip-qsvm

# 5. Generate publication outputs
python report/tables.py
python utils/publication_plots.py

# 6. Verify everything
python scripts/sanity_check.py    # All checks must PASS
pytest tests/ -v                   # All tests must pass
```

## Seed Configuration

All random state is fixed at `SEED = 42`:
- `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`
- SMOTE: `random_state=42`
- StratifiedKFold: `random_state=42`
- XGBoost, LightGBM: `random_state=42`

## Dataset Integrity

SHA-256 checksums are stored in `results/data_hashes.json` after first preprocessing run.

| Dataset | File | Source |
|---------|------|--------|
| UCI CKD | `data/kidney_disease.csv` | Auto-downloaded from UCI ML Repository (id=336) |
| PIMA Diabetes | `data/pima_diabetes.csv` | Auto-downloaded from public mirror (jbrownlee/Datasets) |
| Cleveland Heart | `data/cleveland_heart.csv` | Auto-downloaded from UCI (heart-disease) |
| Framingham HS | `data/framingham.csv` | Manual download from Kaggle |

Verify: `python scripts/sanity_check.py` — Check 2 verifies SHA-256 integrity.

## Model Provenance

After training, `results/provenance_log.json` records:
- SHA-256 of each model checkpoint
- ISO-8601 UTC timestamp
- Python + library versions (torch, pennylane, sklearn, xgboost, lightgbm)
- Combined experiment fingerprint (SHA-256 of all fields)

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | ≥2.1 |
| PennyLane | ≥0.38 |
| scikit-learn | ≥1.4 |
| XGBoost | ≥2.0 |
| LightGBM | ≥4.0 |
| imbalanced-learn | ≥0.12 |

Full pinned versions: `requirements.txt`  
Conda environment: `environment.yml`  
Container: `Dockerfile` (CPU-only)

## Quantum Simulation

All quantum circuits run on `pennylane.device("default.qubit")` — pure-state
classical simulation. No quantum hardware access required. The `default.qubit`
device is deterministic given fixed parameter values.

## Expected Results

After running the full pipeline:

| Model | CKD Acc | CKD AUC | FHS Acc | FHS AUC |
|-------|---------|---------|---------|---------|
| XGBoost | see cv_results.csv | — | see fhs_cv_results.csv | — |
| LightGBM | — | — | — | — |
| MLP | — | — | — | — |
| TabTransformer | — | — | — | — |
| **HybridQT (6q-3L)** | **see cv_results.csv** | — | — | — |

(Exact values depend on hardware due to floating-point ordering; within ±0.5%
as verified by `scripts/sanity_check.py`.)

## Docker Reproducibility

```bash
docker build -t hqct .
docker run -v $(pwd)/data:/workspace/data hqct python main.py --skip-quantum
```
