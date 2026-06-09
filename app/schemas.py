"""
schemas.py
==========
Pydantic input/output models for the Sepsis Prediction API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PatientVitals(BaseModel):
    """
    Raw vitals sent to the /predict endpoint.
    Only the reliable columns that are almost always present.
    The API fills all engineered features (rolling, delta, etc.) with
    safe defaults — the model was trained to handle this via imputation.
    """
    # Core vitals
    HR:     float = Field(..., ge=0, le=300,  description="Heart rate (bpm)")
    O2Sat:  float = Field(..., ge=0, le=100,  description="Oxygen saturation (%)")
    Temp:   float = Field(..., ge=25, le=45,  description="Temperature (°C)")
    SBP:    float = Field(..., ge=0, le=300,  description="Systolic BP (mmHg)")
    MAP:    float = Field(..., ge=0, le=200,  description="Mean arterial pressure (mmHg)")
    DBP:    float = Field(..., ge=0, le=200,  description="Diastolic BP (mmHg)")
    Resp:   float = Field(..., ge=0, le=60,   description="Respiratory rate (breaths/min)")

    # Demographics
    Age:    float = Field(..., ge=0, le=120,  description="Patient age (years)")
    Gender: float = Field(0.0,               description="Gender (0=F, 1=M)")

    # Optional labs — include if available
    Lactate:        Optional[float] = Field(None, description="Lactate (mmol/L)")
    WBC:            Optional[float] = Field(None, description="White blood cell count")
    Creatinine:     Optional[float] = Field(None, description="Creatinine (mg/dL)")
    Platelets:      Optional[float] = Field(None, description="Platelets (x10^9/L)")
    BUN:            Optional[float] = Field(None, description="Blood urea nitrogen")
    Bilirubin_total:Optional[float] = Field(None, description="Total bilirubin (mg/dL)")
    Glucose:        Optional[float] = Field(None, description="Glucose (mg/dL)")
    Temp_history:   Optional[List[float]] = Field(None, description="Last 6 hourly temp readings")
    HR_history:     Optional[List[float]] = Field(None, description="Last 6 hourly HR readings")

    # In PatientVitals, replace the Config class at the bottom with:
model_config = {
    "json_schema_extra": {
        "example": {
            "HR": 115, "O2Sat": 94, "Temp": 38.8,
            "SBP": 88, "MAP": 65, "DBP": 50,
            "Resp": 24, "Age": 67, "Gender": 1,
            "Lactate": 3.2, "WBC": 14.5,
            "Creatinine": 1.8, "Platelets": 120, "BUN": 32,
        }
    }
}

class FeatureExplanation(BaseModel):
    feature:     str
    value:       float
    shap:        float
    explanation: str


class PredictionResponse(BaseModel):
    risk_score:    float = Field(..., description="Probability of sepsis (0–1)")
    prediction:    int   = Field(..., description="1 = sepsis risk, 0 = low risk")
    risk_level:    str   = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    confidence:    str   = Field(..., description="Human-readable confidence statement")
    top_features:  List[FeatureExplanation]
    model_version: str


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    auroc:         float
    recall:        float
    threshold:     float