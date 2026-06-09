"""
main.py
=======
FastAPI application entry point for the Sepsis Early Warning API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from app.schemas import PatientVitals, PredictionResponse, HealthResponse
from app.predict import load_model, predict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Load model once at startup, not on every request ───────────────────────
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model...")
    model_state["model"], model_state["explainer"], model_state["meta"] = load_model()
    log.info(f"Model loaded. AUROC={model_state['meta']['metrics']['auroc']}")
    yield
    model_state.clear()


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Sepsis Early Warning API",
    description = """
Predicts sepsis risk from ICU patient vitals using XGBoost trained on
PhysioNet Challenge 2019 data (40,336 patients).

**AUROC: 0.812 | Recall: 0.75 | Threshold: tuned for clinical recall**

Each prediction includes SHAP-based explanations showing which features
drove the risk score — enabling clinician trust and interpretability.

> ⚠️ For research purposes only. Not a medical device.
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["UI"])
def root():
    """Serve the main HTML interface."""
    html_path = Path(__file__).parent.parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return {"message": "SepsisWatch API", "status": "ok"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    meta    = model_state.get("meta", {})
    metrics = meta.get("metrics", {})
    return {
        "status"       : "ok",
        "model_version": meta.get("model_version", "1.0.0"),
        "auroc"        : metrics.get("auroc", 0.0),
        "recall"       : metrics.get("recall", 0.0),
        "threshold"    : meta.get("threshold", 0.5),
    }

@app.get("/")
def root():
    html_path = Path(__file__).parent.parent / "index.html"

    return {
        "exists": html_path.exists(),
        "path": str(html_path)
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_sepsis(vitals: PatientVitals):
    """
    Predict sepsis risk from patient vitals.

    Returns:
    - **risk_score**: probability 0–1
    - **prediction**: 1 = at risk, 0 = low risk
    - **risk_level**: LOW / MEDIUM / HIGH / CRITICAL
    - **top_features**: top 5 SHAP feature explanations
    """
    if not model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict(
        vitals.model_dump(),
        model_state["model"],
        model_state["explainer"],
        model_state["meta"],
    )
    return result


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(patients: list[PatientVitals]):
    """
    Predict sepsis risk for multiple patients at once.
    Maximum 100 patients per request.
    """
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Max 100 patients per batch")
    if not model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for vitals in patients:
        result = predict(
            vitals.model_dump(),
            model_state["model"],
            model_state["explainer"],
            model_state["meta"],
        )
        results.append(result)
    return {"predictions": results, "count": len(results)}
