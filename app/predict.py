"""
predict.py
==========
Inference logic for the Sepsis Prediction API.

Key design decision:
  The model was trained on 290+ engineered features.
  At inference time we only receive raw vitals (7–15 values).
  This module reconstructs ALL required features from those vitals,
  filling gaps with the training-time medians stored in metadata.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Clinical explanation dictionary ────────────────────────────────────────
FEATURE_EXPLANATIONS = {
    "labs_ordered_3h"       : "High lab ordering rate — clinical concern signal",
    "labs_ordered_this_hour": "Number of labs ordered this hour",
    "sirs_max_so_far"       : "Peak SIRS criteria score during ICU stay",
    "sirs_score"            : "Current SIRS criteria count (0–4)",
    "sirs_2plus"            : "Patient meets ≥2 SIRS criteria",
    "BUN"                   : "Elevated blood urea nitrogen — renal stress",
    "sofa_max_so_far"       : "Worst organ failure score recorded",
    "sofa_proxy"            : "Current organ dysfunction score (0–4)",
    "sofa_critical"         : "Organ failure score ≥ 2 — critical threshold",
    "sofa_renal"            : "Renal dysfunction component of SOFA",
    "sofa_cardiovascular"   : "Cardiovascular dysfunction — low MAP",
    "sofa_coagulation"      : "Coagulation dysfunction — low platelets",
    "sofa_hepatic"          : "Hepatic dysfunction — elevated bilirubin",
    "hours_in_shock"        : "Cumulative hours with haemodynamic instability",
    "shock_index"           : "Heart rate / blood pressure ratio (>1 = danger)",
    "shock_index_high"      : "Shock index above critical threshold (>1.0)",
    "Temp"                  : "Current body temperature",
    "Temp_max_6h"           : "Peak temperature in last 6 hours — fever pattern",
    "Temp_max_3h"           : "Peak temperature in last 3 hours",
    "FiO2"                  : "Fraction of inspired oxygen — respiratory support",
    "FiO2_measured"         : "Supplemental oxygen being administered",
    "Lactate"               : "Blood lactate — metabolic stress indicator",
    "Lactate_min_12h"       : "Lactate trend over 12 hours",
    "Lactate_delta_3h"      : "Lactate change in last 3 hours",
    "Platelets"             : "Low platelets — coagulation dysfunction signal",
    "Resp_mean_12h"         : "Sustained elevated respiratory rate over 12h",
    "Resp_min_12h"          : "Minimum respiratory rate in last 12 hours",
    "Hgb"                   : "Haemoglobin — anaemia can worsen sepsis outcomes",
    "WBC"                   : "White blood cell count — infection response",
    "Alkalinephos"          : "Elevated alkaline phosphatase — liver stress",
    "Glucose_std_12h"       : "Glucose variability over 12 hours",
    "PaCO2"                 : "Arterial CO2 — respiratory compensation",
    "HR_mean_6h"            : "Average heart rate over last 6 hours",
    "HR_std_6h"             : "Heart rate variability — instability signal",
    "MAP_mean_3h"           : "Average mean arterial pressure over 3 hours",
    "pulse_pressure"        : "Systolic minus diastolic BP — narrowing = shock",
    "pulse_pressure_narrow" : "Narrowed pulse pressure — early shock sign",
    "qsofa_score"           : "qSOFA bedside sepsis screening score",
    "qsofa_positive"        : "Positive qSOFA — validated sepsis risk indicator",
    "lactate_x_hr"          : "Lactate × HR interaction — combined stress signal",
    "wbc_x_temp"            : "WBC × temperature — infection severity interaction",
    "creatinine_x_map"      : "Kidney stress relative to perfusion pressure",
    "Potassium"             : "Potassium level — electrolyte balance",
    "PTT"                   : "Partial thromboplastin time — coagulation status",
    "Bilirubin_total"       : "Total bilirubin — liver function marker",
    "ICULOS"                : "Hours since ICU admission",
    "Age"                   : "Patient age — risk factor for sepsis",
}

DEFAULT_EXPLANATION = "Clinical feature contributing to sepsis risk assessment"


# ── Model loader ────────────────────────────────────────────────────────────

_model     = None
_explainer = None
_meta      = None


def load_model():
    global _model, _explainer, _meta
    if _model is None:
        _model     = joblib.load("models/xgb_sepsis.pkl")
        _explainer = joblib.load("models/shap_explainer.pkl")
        _meta      = json.load(open("models/model_metadata.json"))
    return _model, _explainer, _meta


# ── Feature builder ─────────────────────────────────────────────────────────

def _build_feature_row(vitals: dict, feature_cols: list,
                       feature_medians: dict) -> pd.DataFrame:
    """
    Reconstruct the full 290+ feature vector from raw vitals.

    Strategy:
    1. Compute derived features directly from the vitals provided
    2. For engineered features (rolling windows, deltas) use the
       training-time median as a neutral default
    3. This won't be as accurate as having a full patient history,
       but gives a reasonable single-timepoint prediction

    For a production system you'd store patient history in Redis
    and compute real rolling windows — that's a v2 feature.
    """
    # WITH THIS — or() handles None explicitly
    def _safe(val, key, default):
        """Return val if not None, else fall back to median, else default."""
        return val if val is not None else feature_medians.get(key, default)

    hr   = _safe(vitals.get("HR"),              "HR",              80.0)
    sbp  = _safe(vitals.get("SBP"),             "SBP",            120.0)
    dbp  = _safe(vitals.get("DBP"),             "DBP",             80.0)
    map_ = _safe(vitals.get("MAP"),             "MAP",             93.0)
    resp = _safe(vitals.get("Resp"),            "Resp",            16.0)
    temp = _safe(vitals.get("Temp"),            "Temp",            37.0)
    o2   = _safe(vitals.get("O2Sat"),           "O2Sat",           98.0)
    lac  = _safe(vitals.get("Lactate"),         "Lactate",          1.5)
    wbc  = _safe(vitals.get("WBC"),             "WBC",              9.0)
    crea = _safe(vitals.get("Creatinine"),      "Creatinine",       0.9)
    plat = _safe(vitals.get("Platelets"),       "Platelets",      220.0)
    bili = _safe(vitals.get("Bilirubin_total"), "Bilirubin_total",  0.8)
    fio2 = _safe(vitals.get("FiO2"),            "FiO2",            0.21)

    # Directly computable clinical features
    direct = {
        # Raw vitals
        "HR": hr, "O2Sat": o2, "Temp": temp,
        "SBP": sbp, "MAP": map_, "DBP": dbp, "Resp": resp,
        "Age": vitals.get("Age", 60),
        "Gender": vitals.get("Gender", 0),
        "Lactate": lac, "WBC": wbc, "Creatinine": crea,
        "Platelets": plat, "Bilirubin_total": bili,
        "BUN": vitals.get("BUN", feature_medians.get("BUN", 18)),
        "Glucose": vitals.get("Glucose", feature_medians.get("Glucose", 120)),
        "FiO2": fio2,

        # Derived clinical features
        "shock_index"           : hr / (sbp + 1e-5),
        "shock_index_high"      : int(hr / (sbp + 1e-5) > 1.0),
        "pulse_pressure"        : sbp - dbp,
        "pulse_pressure_narrow" : int((sbp - dbp) < 25),
        "sofa_cardiovascular"   : int(map_ < 70),
        "sofa_renal"            : int(crea > 1.2),
        "sofa_coagulation"      : int(plat < 150),
        "sofa_hepatic"          : int(bili > 1.2),
        "sofa_proxy"            : int(map_<70) + int(crea>1.2) + int(plat<150) + int(bili>1.2),
        "sofa_critical"         : int((int(map_<70)+int(crea>1.2)+int(plat<150)+int(bili>1.2)) >= 2),
        "fever"                 : int(temp > 38.3),
        "hypothermia"           : int(temp < 36.0),
        "tachypnea"             : int(resp > 22),
        "tachy_hypo"            : int(hr > 100 and sbp < 90),
        "sirs_score"            : (int(hr>90) + int(resp>22) +
                                   int(temp>38.3 or temp<36.0) +
                                   int(wbc>12 or wbc<4)),
        "sirs_2plus"            : int((int(hr>90)+int(resp>22)+
                                       int(temp>38.3 or temp<36.0)+
                                       int(wbc>12 or wbc<4)) >= 2),
        "qsofa_score"           : int(resp >= 22) + int(sbp <= 100),
        "qsofa_positive"        : int((int(resp>=22) + int(sbp<=100)) >= 2),
        "lactate_x_hr"          : lac * hr,
        "wbc_x_temp"            : wbc * temp,
        "creatinine_x_map"      : crea / (map_ + 1e-5),
        "FiO2_measured"         : int(fio2 > 0.21),

        # Rolling features — use current value as proxy for all windows
        # (best estimate without history)
        "HR_mean_1h": hr, "HR_mean_3h": hr, "HR_mean_6h": hr, "HR_mean_12h": hr,
        "HR_std_1h": 0,   "HR_std_3h": 0,   "HR_std_6h": 0,   "HR_std_12h": 0,
        "HR_min_1h": hr,  "HR_min_3h": hr,  "HR_min_6h": hr,  "HR_min_12h": hr,
        "HR_max_1h": hr,  "HR_max_3h": hr,  "HR_max_6h": hr,  "HR_max_12h": hr,

        "Temp_mean_1h": temp, "Temp_mean_3h": temp, "Temp_mean_6h": temp, "Temp_mean_12h": temp,
        "Temp_std_1h": 0,     "Temp_std_3h": 0,     "Temp_std_6h": 0,     "Temp_std_12h": 0,
        "Temp_min_1h": temp,  "Temp_min_3h": temp,  "Temp_min_6h": temp,  "Temp_min_12h": temp,
        "Temp_max_1h": temp,  "Temp_max_3h": temp,  "Temp_max_6h": temp,  "Temp_max_12h": temp,

        "Resp_mean_1h": resp, "Resp_mean_3h": resp, "Resp_mean_6h": resp, "Resp_mean_12h": resp,
        "Resp_std_1h": 0,     "Resp_std_3h": 0,     "Resp_std_6h": 0,     "Resp_std_12h": 0,
        "Resp_min_1h": resp,  "Resp_min_3h": resp,  "Resp_min_6h": resp,  "Resp_min_12h": resp,
        "Resp_max_1h": resp,  "Resp_max_3h": resp,  "Resp_max_6h": resp,  "Resp_max_12h": resp,

        "SBP_mean_1h": sbp, "SBP_mean_3h": sbp, "SBP_mean_6h": sbp, "SBP_mean_12h": sbp,
        "SBP_std_1h": 0,    "SBP_std_3h": 0,    "SBP_std_6h": 0,    "SBP_std_12h": 0,
        "SBP_min_1h": sbp,  "SBP_min_3h": sbp,  "SBP_min_6h": sbp,  "SBP_min_12h": sbp,
        "SBP_max_1h": sbp,  "SBP_max_3h": sbp,  "SBP_max_6h": sbp,  "SBP_max_12h": sbp,

        "MAP_mean_1h": map_, "MAP_mean_3h": map_, "MAP_mean_6h": map_, "MAP_mean_12h": map_,
        "MAP_std_1h": 0,     "MAP_std_3h": 0,     "MAP_std_6h": 0,     "MAP_std_12h": 0,
        "MAP_min_1h": map_,  "MAP_min_3h": map_,  "MAP_min_6h": map_,  "MAP_min_12h": map_,
        "MAP_max_1h": map_,  "MAP_max_3h": map_,  "MAP_max_6h": map_,  "MAP_max_12h": map_,

        # Delta features — 0 means no change (single timepoint)
        "HR_delta_1h": 0,   "HR_delta_3h": 0,   "HR_delta_6h": 0,
        "Temp_delta_1h": 0, "Temp_delta_3h": 0, "Temp_delta_6h": 0,
        "Resp_delta_1h": 0, "Resp_delta_3h": 0, "Resp_delta_6h": 0,
        "SBP_delta_1h": 0,  "SBP_delta_3h": 0,  "SBP_delta_6h": 0,
        "MAP_delta_1h": 0,  "MAP_delta_3h": 0,  "MAP_delta_6h": 0,

        # Expanding features — use current as proxy
        "sofa_max_so_far" : int(map_<70)+int(crea>1.2)+int(plat<150)+int(bili>1.2),
        "sirs_max_so_far" : int(hr>90)+int(resp>22)+int(temp>38.3 or temp<36.0)+int(wbc>12 or wbc<4),
        "hours_in_shock"  : int(hr / (sbp + 1e-5) > 1.0),
    }

    # Build full row — start with medians for everything, then override with computed
    row = {col: feature_medians.get(col, 0.0) for col in feature_cols}
    row.update(direct)

    result = pd.DataFrame([row])[feature_cols]
    result = result.astype(float)   # force all columns to float64 — XGBoost requires this
    return result


# ── Risk level helper ───────────────────────────────────────────────────────

def _risk_level(score: float) -> tuple[str, str]:
    if score >= 0.75:
        return "CRITICAL", f"Very high sepsis risk ({score*100:.0f}%). Immediate clinical review recommended."
    elif score >= 0.50:
        return "HIGH",     f"High sepsis risk ({score*100:.0f}%). Close monitoring and lab workup advised."
    elif score >= 0.30:
        return "MEDIUM",   f"Moderate sepsis risk ({score*100:.0f}%). Continue monitoring vital signs."
    else:
        return "LOW",      f"Low sepsis risk ({score*100:.0f}%). Routine monitoring."


# ── Main predict function ───────────────────────────────────────────────────

def predict(vitals: dict, model, explainer, meta: dict) -> dict:
    feature_cols    = meta["feature_cols"]
    threshold       = meta["threshold"]
    feature_medians = meta.get("feature_medians", {})

    # Build feature row
    row = _build_feature_row(vitals, feature_cols, feature_medians)

    # Predict probability
    proba      = float(model.predict_proba(row)[0, 1])
    prediction = int(proba >= threshold)

    # SHAP explanation
    shap_vals = explainer.shap_values(row)[0]
    import math

    # Replace nan/inf shap values with 0.0 before sorting
    clean_shap = [v if (v == v and not math.isinf(v)) else 0.0 for v in shap_vals]

    top_pairs = sorted(zip(feature_cols, clean_shap, row.iloc[0].values),
                       key=lambda x: abs(x[1]), reverse=True)[:5]

    top_features = [
        {
            "feature"    : feat,
            "value"      : round(float(val) if not math.isnan(float(val)) else 0.0, 3),
            "shap"       : round(float(shap), 4),
            "explanation": FEATURE_EXPLANATIONS.get(feat, DEFAULT_EXPLANATION),
        }
        for feat, shap, val in top_pairs
    ]
    risk_level, confidence = _risk_level(proba)

    return {
        "risk_score"   : round(proba, 4),
        "prediction"   : prediction,
        "risk_level"   : risk_level,
        "confidence"   : confidence,
        "top_features" : top_features,
        "model_version": meta.get("model_version", "1.0.0"),
    }