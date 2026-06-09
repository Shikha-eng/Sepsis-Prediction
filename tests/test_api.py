"""
test_api.py
===========
Tests for the Sepsis Prediction API.
Run with: pytest tests/test_api.py -v

Requires the API to be running:
    uvicorn app.main:app --reload
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app, model_state
from app.predict import load_model

# Manually load model before tests run — bypasses lifespan
@pytest.fixture(scope="session", autouse=True)
def load_model_for_tests():
    model_state["model"], model_state["explainer"], model_state["meta"] = load_model()
    yield
    model_state.clear()

client = TestClient(app)


# ── Fixtures ────────────────────────────────────────────────────────────────

HIGH_RISK_PATIENT = {
    "HR": 118, "O2Sat": 92, "Temp": 39.2,
    "SBP": 85,  "MAP": 62,  "DBP": 48,
    "Resp": 26, "Age": 72,  "Gender": 1,
    "Lactate": 4.1, "WBC": 16.2, "Creatinine": 2.1,
    "Platelets": 95, "BUN": 48,
}

LOW_RISK_PATIENT = {
    "HR": 72,  "O2Sat": 98, "Temp": 36.8,
    "SBP": 122, "MAP": 88,  "DBP": 75,
    "Resp": 14, "Age": 45,  "Gender": 0,
}


# ── Tests ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "auroc" in data
    assert data["auroc"] > 0.75


def test_predict_returns_correct_fields():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    assert r.status_code == 200
    data = r.json()
    assert "risk_score"   in data
    assert "prediction"   in data
    assert "risk_level"   in data
    assert "confidence"   in data
    assert "top_features" in data
    assert len(data["top_features"]) == 5


def test_predict_risk_score_range():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    score = r.json()["risk_score"]
    assert 0.0 <= score <= 1.0


def test_high_risk_patient_flagged():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    data = r.json()
    assert data["risk_level"] in ["HIGH", "CRITICAL"]


def test_low_risk_patient_not_critical():
    r = client.post("/predict", json=LOW_RISK_PATIENT)
    data = r.json()
    assert data["risk_level"] in ["LOW", "MEDIUM"]


def test_top_features_have_explanation():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    for feat in r.json()["top_features"]:
        assert "explanation" in feat
        assert len(feat["explanation"]) > 5


def test_missing_required_field():
    bad_patient = {"HR": 110}   # missing most fields
    r = client.post("/predict", json=bad_patient)
    assert r.status_code == 422   # Pydantic validation error


def test_out_of_range_value():
    bad_patient = {**LOW_RISK_PATIENT, "HR": 999}   # HR > 300
    r = client.post("/predict", json=bad_patient)
    assert r.status_code == 422


def test_batch_predict():
    r = client.post("/predict/batch", json=[HIGH_RISK_PATIENT, LOW_RISK_PATIENT])
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_batch_limit():
    patients = [LOW_RISK_PATIENT] * 101
    r = client.post("/predict/batch", json=patients)
    assert r.status_code == 400