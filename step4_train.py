"""
Step 4: Model Training
Trains RF and XGBoost for both CYP11B2 and CYP11B1 using:
  - Scaffold split (primary)
  - Random split (comparison baseline)
  - Feature set: ECFP4 (primary)

Inputs:  data_cleaned.csv
         features/X_ecfp4.npy
         splits/scaffold_split.npz
         splits/random_split.npz

Outputs: models/rf_b2_scaffold.pkl
         models/xgb_b2_scaffold.pkl
         models/rf_b1_scaffold.pkl
         models/xgb_b1_scaffold.pkl
         models/rf_b2_random.pkl
         models/xgb_b2_random.pkl
         models/rf_b1_random.pkl
         models/xgb_b1_random.pkl
         models/split_info.npz  (stores all split index arrays)
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

os.makedirs("models", exist_ok=True)

# ── Load data and features ────────────────────────────────────────────────────
df     = pd.read_csv("data_cleaned.csv")
X_fp   = np.load("features/X_ecfp4.npy")

scaffold = np.load("splits/scaffold_split.npz")
random   = np.load("splits/random_split.npz")

splits = {
    "scaffold": {k: scaffold[k] for k in scaffold.files},
    "random":   {k: random[k]   for k in random.files},
}

# ── Helper: build train/val/test arrays for one target & split ────────────────
def get_xy(X_all, y_series, train_idx, val_idx, test_idx):
    """Filter NaN targets and remap indices."""
    mask = y_series.notna().values

    # Keep only indices where target is valid
    tr = [i for i in train_idx if mask[i]]
    vl = [i for i in val_idx   if mask[i]]
    te = [i for i in test_idx  if mask[i]]

    # Map original row indices → compressed (NaN-filtered) indices
    pos = np.where(mask)[0]
    p2n = {old: new for new, old in enumerate(pos)}

    X_tr = X_all[[p2n[i] for i in tr]]
    y_tr = y_series.dropna().values[[p2n[i] for i in tr]]
    X_vl = X_all[[p2n[i] for i in vl]]
    y_vl = y_series.dropna().values[[p2n[i] for i in vl]]
    X_te = X_all[[p2n[i] for i in te]]
    y_te = y_series.dropna().values[[p2n[i] for i in te]]

    return (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te)

# ── Training loop ─────────────────────────────────────────────────────────────
results = {}

for target in ["pIC50_B2", "pIC50_B1"]:
    tag = "b2" if "B2" in target else "b1"
    y = df[target]

    for split_name, sp in splits.items():
        print(f"\n{'='*60}")
        print(f"Target={target}  Split={split_name}")

        (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = get_xy(
            X_fp, y, sp["train_idx"], sp["val_idx"], sp["test_idx"]
        )
        print(f"  Train={len(y_tr)}, Val={len(y_vl)}, Test={len(y_te)}")

        # ── Random Forest ──────────────────────────────────────────────────
        rf = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_tr, y_tr)

        y_pred_vl = rf.predict(X_vl)
        y_pred_te = rf.predict(X_te)

        r2_vl  = r2_score(y_vl, y_pred_vl)
        r2_te  = r2_score(y_te, y_pred_te)
        rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))

        print(f"  [RF]  Val R²={r2_vl:.3f}  |  Test R²={r2_te:.3f}  RMSE={rmse_te:.3f}")

        rf_path = f"models/rf_{tag}_{split_name}.pkl"
        joblib.dump(rf, rf_path)
        results[f"RF_{tag}_{split_name}"] = {
            "val_r2": r2_vl, "test_r2": r2_te, "test_rmse": rmse_te,
            "y_test": y_te, "y_pred": y_pred_te
        }

        # ── XGBoost ────────────────────────────────────────────────────────
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred_vl_xgb = xgb_model.predict(X_vl)
        y_pred_te_xgb = xgb_model.predict(X_te)

        r2_vl_xgb   = r2_score(y_vl, y_pred_vl_xgb)
        r2_te_xgb   = r2_score(y_te, y_pred_te_xgb)
        rmse_te_xgb = np.sqrt(mean_squared_error(y_te, y_pred_te_xgb))

        print(f"  [XGB] Val R²={r2_vl_xgb:.3f}  |  Test R²={r2_te_xgb:.3f}  RMSE={rmse_te_xgb:.3f}")

        xgb_path = f"models/xgb_{tag}_{split_name}.pkl"
        joblib.dump(xgb_model, xgb_path)
        results[f"XGB_{tag}_{split_name}"] = {
            "val_r2": r2_vl_xgb, "test_r2": r2_te_xgb, "test_rmse": rmse_te_xgb,
            "y_test": y_te, "y_pred": y_pred_te_xgb
        }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("模型对比汇总（Test set）:")
print(f"{'模型':<30} {'Val R²':>8} {'Test R²':>8} {'Test RMSE':>10}")
print("-" * 60)
for name, m in results.items():
    print(f"{name:<30} {m['val_r2']:>8.3f} {m['test_r2']:>8.3f} {m['test_rmse']:>10.3f}")

# Save predictions for evaluation script
joblib.dump(results, "models/train_results.pkl")
print("\n已保存训练结果到 models/train_results.pkl")
print("已保存所有模型到 models/ 目录")
