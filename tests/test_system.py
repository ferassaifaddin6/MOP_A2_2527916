"""Happy-path AND failure-path tests."""
import copy
import hashlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import app
from src.data_generation import write_dataset
from src.inference import load_bundle, predict_record
from src.validation import DataValidationError, assert_no_leakage, clean, parse_identity, validate_raw
from tests.conftest import SEED, STUDENT_ID, VALID


def test_valid_inference(trained):
    out = predict_record(VALID, load_bundle())
    assert out["verification_code"] == f"AI-A2-{SEED:04d}"
    assert out["predicted_yield_tons"] > 0
    assert out["high_risk"] in (0, 1) and 0 <= out["probability_high_risk"] <= 1
    assert isinstance(out["cluster"], int)


def test_missing_field_rejected(trained):
    bad = copy.deepcopy(VALID); del bad["rainfall_mm"]
    with pytest.raises(ValidationError):
        predict_record(bad, load_bundle())


def test_wrong_type_rejected(trained):
    bad = {**VALID, "avg_temp_c": "hot"}
    with pytest.raises(ValidationError):
        predict_record(bad, load_bundle())


def test_range_error_rejected(trained):
    with pytest.raises(ValidationError):
        predict_record({**VALID, "rainfall_mm": 5000.0}, load_bundle())
    with pytest.raises(ValidationError):
        predict_record({**VALID, "season": "C"}, load_bundle())


def test_reproducible_fingerprint(tmp_path):
    h1 = write_dataset(SEED, tmp_path / "a.csv", tmp_path / "a.txt", "x")
    h2 = write_dataset(SEED, tmp_path / "b.csv", tmp_path / "b.txt", "x")
    assert h1 == h2 == hashlib.sha256((tmp_path / "a.csv").read_bytes()).hexdigest()
    assert len(pd.read_csv(tmp_path / "a.csv")) == 240 + SEED % 41 + 3


def test_wrong_seed_rejected():
    with pytest.raises(DataValidationError):
        parse_identity(STUDENT_ID, SEED + 1)


def test_data_problems_detected_and_handled(tmp_path):
    write_dataset(SEED, tmp_path / "d.csv", tmp_path / "f.txt", "x")
    df = pd.read_csv(tmp_path / "d.csv")
    report = validate_raw(df)
    assert report["duplicate_rows"] == 3 and report["missing_values"]
    assert len(clean(df)) == len(df) - 3


def test_schema_and_leakage_checks(tmp_path):
    with pytest.raises(DataValidationError):
        assert_no_leakage(["rainfall_mm", "yield_tons"])
    write_dataset(SEED, tmp_path / "d.csv", tmp_path / "f.txt", "x")
    df = pd.read_csv(tmp_path / "d.csv")
    with pytest.raises(DataValidationError):
        validate_raw(df.drop(columns=["season"]))
    df.loc[0, "rainfall_mm"] = 99999
    with pytest.raises(DataValidationError):
        validate_raw(df)


def test_api_health_info_predict(trained):
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/model-info").json()["verification_code"] == f"AI-A2-{SEED:04d}"
    r = c.post("/predict", json=VALID)
    assert r.status_code == 200 and "predicted_yield_tons" in r.json()


def test_api_invalid_returns_422(trained):
    c = TestClient(app)
    assert c.post("/predict", json={**VALID, "fertilizer_kg": -5}).status_code == 422
    assert c.post("/predict", json={"season": "A"}).status_code == 422
