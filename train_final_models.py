"""
Train final SVR models on ALL available data (no split).

Feature pipeline:
  X_combined (415, 2189)
      → feature_selector  (IndexSelector, saved separately)
      → X_selected (415, 300)
      → final_model_B2 / final_model_B1
            Pipeline(StandardScaler → SVR with best Optuna params)

Best Optuna params (from stratified-scaffold CV on X_selected):
  B2: C=3.41,  epsilon=0.112, gamma=0.00144
  B1: C=639,   epsilon=0.28,  gamma=0.00124

Inputs:  data_cleaned.csv
         features/X_combined.npy          (415, 2189)
         features/selected_feature_idx.npy

Outputs: models/feature_selector.pkl      — IndexSelector (X_combined → X_selected)
         models/final_model_B2.pkl        — Pipeline(StandardScaler + SVR) for CYP11B2
         models/final_model_B1.pkl        — Pipeline(StandardScaler + SVR) for CYP11B1
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

os.makedirs("models", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df  = pd.read_csv("data_cleaned.csv")
X_combined = np.load("features/X_combined.npy")          # (415, 2189)
selected_idx = np.load("features/selected_feature_idx.npy")  # (300,)

print(f"X_combined shape : {X_combined.shape}")
print(f"Selected indices : {len(selected_idx)} features")

# ── Feature selector ───────────────────────────────────────────────────────────
class IndexSelector(BaseEstimator, TransformerMixin):
    """Select columns from X_combined by pre-computed indices."""
    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices]

feature_selector = IndexSelector(indices=selected_idx)
joblib.dump(feature_selector, "models/feature_selector.pkl")
print("Saved models/feature_selector.pkl")

# Verify selector output matches X_selected
X_selected_check = np.load("features/X_selected.npy")
X_via_selector   = feature_selector.transform(X_combined)
assert np.allclose(X_via_selector, X_selected_check), \
    "feature_selector output does not match X_selected.npy — check indices"
print("  Verified: feature_selector.transform(X_combined) == X_selected.npy ✓")

# ── Best Optuna hyperparams ────────────────────────────────────────────────────
BEST_PARAMS = {
    "pIC50_B2": dict(C=3.41,  epsilon=0.112, gamma=0.00144),
    "pIC50_B1": dict(C=639.0, epsilon=0.28,  gamma=0.00124),
}

MODEL_NAMES = {
    "pIC50_B2": "final_model_B2",
    "pIC50_B1": "final_model_B1",
}

# ── Train on ALL valid samples ────────────────────────────────────────────────
X_selected = feature_selector.transform(X_combined)   # (415, 300)

print()
for target, out_name in MODEL_NAMES.items():
    y_series = df[target]
    mask     = y_series.notna().values
    y_all    = y_series.dropna().values
    X_all    = X_selected[mask]
    n        = len(y_all)

    params = BEST_PARAMS[target]
    svr    = SVR(kernel="rbf", **params)
    model  = Pipeline([("scaler", StandardScaler()), ("svr", svr)])
    model.fit(X_all, y_all)

    out_path = f"models/{out_name}.pkl"
    joblib.dump(model, out_path)

    # Sanity check: train-set R² (should be high — model fitted on same data)
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_all, model.predict(X_all))

    print(f"[{target}]  n={n}  params: C={params['C']}, "
          f"ε={params['epsilon']}, γ={params['gamma']}")
    print(f"  Train R² (in-sample): {r2_train:.4f}")
    print(f"  Saved → {out_path}")
    print()

# ── Usage note ─────────────────────────────────────────────────────────────────
print("="*55)
print("Usage for new compounds:")
print("  selector = joblib.load('models/feature_selector.pkl')")
print("  model_b2 = joblib.load('models/final_model_B2.pkl')")
print("  model_b1 = joblib.load('models/final_model_B1.pkl')")
print("  X_new_sel = selector.transform(X_new_combined)")
print("  pred_b2   = model_b2.predict(X_new_sel)")
print("  pred_b1   = model_b1.predict(X_new_sel)")
