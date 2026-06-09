"""
save_metadata.py
================
Run this ONCE after training to update model_metadata.json
with the feature medians needed for inference-time imputation.

Run:
    python save_metadata.py
"""

import json
import pandas as pd
import numpy as np

FEATURES_PATH = "data/features.parquet"
META_PATH     = "models/model_metadata.json"

NON_FEATURE = ['SepsisLabel','Patient_ID','Hour','ICULOS','Unnamed: 0']

print("Loading features...")
df = pd.read_parquet(FEATURES_PATH)
feature_cols = [c for c in df.columns if c not in NON_FEATURE]

print("Computing medians...")
medians = df[feature_cols].median().to_dict()
medians = {k: round(float(v), 6) for k, v in medians.items()}

print("Updating metadata...")
with open(META_PATH, "r") as f:
    meta = json.load(f)

meta["feature_medians"] = medians

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved {len(medians)} feature medians to {META_PATH}")
print("Done. You can now build the Docker image.")