# Contributing to HQCT

Thank you for your interest in the Hybrid Quantum-Classical Transformer project.

## Development Setup

```bash
conda env create -f environment.yml
conda activate hqct
pip install -e .
```

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

All 5 test modules must pass before submitting a PR.

## Code Style

- Python 3.10+, type hints preferred.
- Formatter: `black` (line length 99); imports: `isort`.
- No bare `except:` — always catch specific exceptions.
- No `print()` in library code — use `utils/logging_setup.py` logger.

## Adding a New Model

1. Implement in `models/your_model.py` with `train_your_model()` function.
2. Add 10-fold CV function `cv_your_model()` to `cv_evaluation.py` and `fhs_cv_evaluation.py`.
3. Add to `models/baselines.py::run_training()`.
4. Add unit test in `tests/test_your_model.py`.
5. Run `python scripts/sanity_check.py` — all checks must PASS.

## Quantum Circuit Changes

- VQC changes must be backward-compatible: keep `LEGACY_QC_CFG` intact.
- New circuit configs must pass `tests/test_quantum_layer.py`.
- Update `results/latex_tables/table3_quantum_circuit.tex` with new expressibility values.

## Provenance

Every training run must produce valid provenance records:
- `results/provenance_log.json` — model SHA-256 fingerprints.
- `results/data_hashes.json` — dataset SHA-256 fingerprints.

Run `python scripts/sanity_check.py` to verify.

## Pull Request Checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Sanity check passes: `python scripts/sanity_check.py`
- [ ] LaTeX tables regenerate: `python report/tables.py`
- [ ] `requirements.txt` updated if new dependencies added
- [ ] No hardcoded paths or API keys in code
