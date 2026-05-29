"""
tests/test_integrity.py -- Unit tests for utils/integrity.py.

Tests: compute_sha256, verify_sha256 (tamper detection), sign_model structure,
log_data_hash, save_provenance_record.
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "test_data.bin"
    f.write_bytes(b"Hello, HQCT provenance system!" * 100)
    return f


def test_compute_sha256_returns_hex(tmp_file):
    from utils.integrity import compute_sha256
    digest = compute_sha256(str(tmp_file))
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_compute_sha256_deterministic(tmp_file):
    from utils.integrity import compute_sha256
    d1 = compute_sha256(str(tmp_file))
    d2 = compute_sha256(str(tmp_file))
    assert d1 == d2


def test_verify_sha256_passes_on_match(tmp_file):
    from utils.integrity import compute_sha256, verify_sha256
    digest = compute_sha256(str(tmp_file))
    assert verify_sha256(str(tmp_file), digest) is True


def test_verify_sha256_raises_on_tamper(tmp_file):
    from utils.integrity import IntegrityError, verify_sha256
    with pytest.raises(IntegrityError):
        verify_sha256(str(tmp_file), "a" * 64)


def test_sign_model_structure(tmp_file):
    from utils.integrity import sign_model
    record = sign_model(str(tmp_file), {"model": "test_model", "dataset": "test"})
    assert "timestamp_utc" in record
    assert "model_sha256" in record
    assert "experiment_fingerprint" in record
    assert "library_versions" in record
    assert len(record["model_sha256"]) == 64
    assert len(record["experiment_fingerprint"]) == 64


def test_sign_model_sha256_matches_file(tmp_file):
    from utils.integrity import compute_sha256, sign_model
    expected = compute_sha256(str(tmp_file))
    record = sign_model(str(tmp_file), {})
    assert record["model_sha256"] == expected


def test_save_and_load_provenance(tmp_path, tmp_file):
    from utils.integrity import sign_model, save_provenance_record
    record = sign_model(str(tmp_file), {"test": True})
    log_path = tmp_path / "provenance_log.json"
    save_provenance_record(record, str(log_path))
    assert log_path.exists()
    with open(log_path) as f:
        log = json.load(f)
    assert isinstance(log, list)
    assert log[0]["model_sha256"] == record["model_sha256"]


def test_save_provenance_appends(tmp_path, tmp_file):
    from utils.integrity import sign_model, save_provenance_record
    log_path = tmp_path / "provenance_log.json"
    r1 = sign_model(str(tmp_file), {"seq": 1})
    r2 = sign_model(str(tmp_file), {"seq": 2})
    save_provenance_record(r1, str(log_path))
    save_provenance_record(r2, str(log_path))
    with open(log_path) as f:
        log = json.load(f)
    assert len(log) == 2


def test_log_data_hash(tmp_path, tmp_file):
    from utils.integrity import log_data_hash
    registry_path = tmp_path / "data_hashes.json"
    digest = log_data_hash(str(tmp_file), str(registry_path), label="test_dataset")
    assert len(digest) == 64
    assert registry_path.exists()
    with open(registry_path) as f:
        reg = json.load(f)
    assert "test_dataset" in reg
    assert reg["test_dataset"]["sha256"] == digest
