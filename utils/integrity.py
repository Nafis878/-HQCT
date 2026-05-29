"""
utils/integrity.py -- SHA-256 cryptographic provenance for datasets and models.

Ensures full reproducibility and tamper-evidence required by Q1 journals.
Every dataset and model checkpoint is fingerprinted; a combined experiment
fingerprint ties all artifacts to a single verifiable record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class IntegrityError(Exception):
    """Raised when a SHA-256 verification fails."""


# ── Core SHA-256 helpers ──────────────────────────────────────────────────────

def compute_sha256(filepath: str) -> str:
    """Stream-hash a file in 64 KB chunks; return lowercase hex digest."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(filepath: str, expected: str) -> bool:
    """Return True if file matches expected hex digest; raise IntegrityError on mismatch."""
    actual = compute_sha256(filepath)
    if actual != expected.lower():
        raise IntegrityError(
            f"SHA-256 mismatch for {filepath}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
    return True


def _lib_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not_installed"


# ── Model provenance ──────────────────────────────────────────────────────────

def sign_model(
    model_path: str,
    metadata: dict,
    data_sha256: Optional[str] = None,
) -> dict:
    """
    Produce a JSON provenance record for a saved model checkpoint.

    Parameters
    ----------
    model_path   : path to the saved weights file
    metadata     : arbitrary key/value pairs (model name, config, etc.)
    data_sha256  : SHA-256 of the training dataset used (optional)

    Returns a dict that can be appended to results/provenance_log.json.
    """
    model_sha256 = compute_sha256(model_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp_utc": timestamp,
        "model_path": str(Path(model_path).name),
        "model_sha256": model_sha256,
        "data_sha256": data_sha256 or "not_provided",
        "python_version": sys.version,
        "platform": platform.platform(),
        "library_versions": {
            "torch":        _lib_version("torch"),
            "pennylane":    _lib_version("pennylane"),
            "sklearn":      _lib_version("sklearn"),
            "xgboost":      _lib_version("xgboost"),
            "lightgbm":     _lib_version("lightgbm"),
            "numpy":        _lib_version("numpy"),
            "pandas":       _lib_version("pandas"),
        },
        "random_seeds": metadata.get("random_seeds", {"global": 42}),
        "metadata": metadata,
    }

    # Combined experiment fingerprint
    fingerprint_src = (
        record["model_sha256"]
        + record["data_sha256"]
        + record["timestamp_utc"]
        + record["python_version"]
        + json.dumps(record["library_versions"], sort_keys=True)
    )
    record["experiment_fingerprint"] = hashlib.sha256(
        fingerprint_src.encode()
    ).hexdigest()

    return record


def save_provenance_record(record: dict, out_path: str) -> None:
    """Append a provenance record to a JSON log file (creates if absent)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except json.JSONDecodeError:
            existing = []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))


def load_and_verify_provenance(
    record_path: str,
    model_path: str,
    data_path: Optional[str] = None,
) -> bool:
    """
    Load a provenance log, find the entry for model_path, and verify SHA-256.

    Returns True on success; warns (but does not raise) if no record found
    for backward compatibility with pre-provenance checkpoints.
    """
    p = Path(record_path)
    if not p.exists():
        print(f"  [integrity] WARNING: No provenance log at {record_path}")
        return False

    records = json.loads(p.read_text())
    model_name = Path(model_path).name
    match = next(
        (r for r in records if r.get("model_path") == model_name), None
    )
    if match is None:
        print(f"  [integrity] WARNING: No provenance record for {model_name}")
        return False

    try:
        verify_sha256(model_path, match["model_sha256"])
        print(f"  [integrity] OK: {model_name} matches provenance record.")
        if data_path and match.get("data_sha256") != "not_provided":
            verify_sha256(data_path, match["data_sha256"])
            print(f"  [integrity] OK: dataset SHA-256 verified.")
        return True
    except IntegrityError as e:
        print(f"  [integrity] TAMPER DETECTED: {e}")
        return False


# ── Dataset hash registry ─────────────────────────────────────────────────────

def log_data_hash(filepath: str, registry_path: str, label: str = "") -> str:
    """
    Compute SHA-256 of filepath and persist to a JSON registry.
    Returns the hex digest.
    """
    digest = compute_sha256(filepath)
    p = Path(registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    registry: dict = {}
    if p.exists():
        try:
            registry = json.loads(p.read_text())
        except json.JSONDecodeError:
            registry = {}
    key = label or Path(filepath).name
    registry[key] = {
        "path": str(filepath),
        "sha256": digest,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(registry, indent=2))
    return digest


# ── Experiment manifest ───────────────────────────────────────────────────────

def save_manifest(results_dir: str, cfg_dict: dict) -> None:
    """
    Write results/experiment_manifest.json listing every file in results_dir,
    its SHA-256, and the config snapshot. Called at end of each pipeline run.
    """
    results = Path(results_dir)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg_dict,
        "artifacts": {},
    }
    for f in sorted(results.rglob("*")):
        if f.is_file() and f.suffix not in (".json",):
            try:
                manifest["artifacts"][str(f.relative_to(results))] = compute_sha256(str(f))
            except (PermissionError, OSError):
                pass
    out = results / "experiment_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"  [integrity] Manifest saved: {out}")
