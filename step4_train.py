"""
Step 4: Model Training
Trains RF, XGBoost, and SVR for both CYP11B2 and CYP11B1 using:
  - Stratified-scaffold split (primary — recommended)
  - Scaffold split (hard external validation)
  - Random split (comparison baseline)
  - Feature set: X_combined (ECFP4 + RDKit descriptors, 2189 dims)

Models are saved as sklearn Pipelines (StandardScaler + model) so they
can be used directly without a separate scaler.

Inputs:  data_cleaned.csv
         features/X_combined.npy
         splits/stratified_scaffold_split.npz
         splits/scaffold_split.npz
         splits/random_split.npz

Outputs: models/{rf,xgb,svr}_{b2,b1}_{stratified_scaffold,scaffold,random}.pkl
         models/train_results.pkl
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── Load data and features ────────────────────────────────────────────────────
df   = pd.read_csv("data_cleaned.csv")
X_fp = np.load("features/X_combined.npy")  # (415, 2189)

strat  = np.load("splits/stratified_scaffold_split.npz")
scaff  = np.load("splits/scaffold_split.npz")
rand   = np.load("splits/random_split.npz")

splits = {
    "stratified_scaffold": {k: strat[k] for k in strat.files},
    "scaffold":            {k: scaff[k] for k in scaff.files},
    "random":              {k: rand[k]  for k in rand.files},
}

# ── Helper: build train/val/test arrays for one target & split ────────────────
def get_xy(X_all, y_series, train_idx, val_idx, test_idx):
    """Filter NaN targets and return (X, y) tuples for train/val/test."""
    mask = y_series.notna().values
    tr = [i for i in train_idx if mask[i]]
    vl = [i for i in val_idx   if mask[i]]
    te = [i for i in test_idx  if mask[i]]

    pos  = np.where(mask)[0]
    p2n  = {old: new for new, old in enumerate(pos)}
    yval = y_series.dropna().values

    X_tr = X_all[[p2n[i] for i in tr]]; y_tr = yval[[p2n[i] for i in tr]]
    X_vl = X_all[[p2n[i] for i in vl]]; y_vl = yval[[p2n[i] for i in vl]]
    X_te = X_all[[p2n[i] for i in te]]; y_te = yval[[p2n[i] for i in te]]
    return (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te)

# ── SVR tuning via Optuna ─────────────────────────────────────────────────────
def tune_svr(X_tr_sc, y_tr, X_vl_sc, y_vl, n_trials=60):
    """Search SVR hyperparams using the val set. Returns best SVR (unfitted)."""
    def objective(trial):
        C   = trial.suggest_float("C",       1e-1, 1e3,  log=True)
        eps = trial.suggest_float("epsilon", 0.01, 0.5)
        g   = trial.suggest_float("gamma",   1e-4, 1e-1, log=True)
        model = SVR(kernel="rbf", C=C, epsilon=eps, gamma=g)
        model.fit(X_tr_sc, y_tr)
        return -r2_score(y_vl, model.predict(X_vl_sc))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"    Best SVR: C={bp['C']:.3g}, ε={bp['epsilon']:.3g}, γ={bp['gamma']:.3g}")
    return SVR(kernel="rbf", C=bp["C"], epsilon=bp["epsilon"], gamma=bp["gamma"])

# ── 5-fold CV on full dataset (primary metric for small datasets) ─────────────
def cv_r2(pipeline_factory, X_all, y_all, n_splits=5):
    """5-fold random CV R² on all valid samples."""
    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr_i, te_i in kf.split(X_all):
        p = pipeline_factory()
        p.fit(X_all[tr_i], y_all[tr_i])
        scores.append(r2_score(y_all[te_i], p.predict(X_all[te_i])))
    return np.array(scores)

# ── Training loop ─────────────────────────────────────────────────────────────
results = {}

for target in ["pIC50_B2", "pIC50_B1"]:
    tag = "b2" if "B2" in target else "b1"
    y = df[target]

    # ── 5-fold CV (primary metric: uses all valid samples, random folds) ──────
    y_all_valid = y.dropna().values
    X_all_valid = X_fp[y.notna().values]
    print(f"\n[CV] {target}  n={len(y_all_valid)}")

    cv_rf  = cv_r2(lambda: Pipeline([("sc", StandardScaler()),
                                      ("rf", RandomForestRegressor(n_estimators=500, max_features=0.3,
                                                                    min_samples_leaf=3, n_jobs=-1, random_state=42))]),
                   X_all_valid, y_all_valid)
    cv_xgb = cv_r2(lambda: Pipeline([("sc", StandardScaler()),
                                      ("xgb", xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                                                 max_depth=4, subsample=0.8, colsample_bytree=0.6,
                                                                 reg_alpha=0.5, reg_lambda=5.0, min_child_weight=5,
                                                                 verbosity=0, n_jobs=-1, random_state=42))]),
                   X_all_valid, y_all_valid)
    cv_svr = cv_r2(lambda: Pipeline([("sc", StandardScaler()),
                                      ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"))]),
                   X_all_valid, y_all_valid)

    print(f"  RF  5-fold CV R²: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")
    print(f"  XGB 5-fold CV R²: {cv_xgb.mean():.3f} ± {cv_xgb.std():.3f}")
    print(f"  SVR 5-fold CV R²: {cv_svr.mean():.3f} ± {cv_svr.std():.3f}")

    results[f"RF_{tag}_cv"]  = {"cv_r2_mean": cv_rf.mean(),  "cv_r2_std": cv_rf.std()}
    results[f"XGB_{tag}_cv"] = {"cv_r2_mean": cv_xgb.mean(), "cv_r2_std": cv_xgb.std()}
    results[f"SVR_{tag}_cv"] = {"cv_r2_mean": cv_svr.mean(), "cv_r2_std": cv_svr.std()}

    for split_name, sp in splits.items():
        print(f"\n{'='*60}")
        print(f"Target={target}  Split={split_name}")

        (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = get_xy(
            X_fp, y, sp["train_idx"], sp["val_idx"], sp["test_idx"]
        )
        print(f"  Train={len(y_tr)}, Val={len(y_vl)}, Test={len(y_te)}")

        # Pre-scale once (used by SVR tuning; RF/XGB pipelines scale internally)
        scaler_tmp = StandardScaler().fit(X_tr)
        X_tr_sc = scaler_tmp.transform(X_tr)
        X_vl_sc = scaler_tmp.transform(X_vl)

        # ── Random Forest ──────────────────────────────────────────────────
        rf_core = RandomForestRegressor(
            n_estimators=500,
            max_features=0.3,       # 30% features — better for small-data extrapolation
            min_samples_leaf=3,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )
        rf_pipe = Pipeline([("scaler", StandardScaler()), ("rf", rf_core)])
        rf_pipe.fit(X_tr, y_tr)

        r2_vl  = r2_score(y_vl,  rf_pipe.predict(X_vl))
        r2_te  = r2_score(y_te,  rf_pipe.predict(X_te))
        rmse_te = np.sqrt(mean_squared_error(y_te, rf_pipe.predict(X_te)))
        print(f"  [RF]  Val R²={r2_vl:.3f}  |  Test R²={r2_te:.3f}  RMSE={rmse_te:.3f}")

        joblib.dump(rf_pipe, f"models/rf_{tag}_{split_name}.pkl")
        results[f"RF_{tag}_{split_name}"] = {
            "val_r2": r2_vl, "test_r2": r2_te, "test_rmse": rmse_te,
            "y_test": y_te, "y_pred": rf_pipe.predict(X_te)
        }

        # ── XGBoost ────────────────────────────────────────────────────────
        xgb_core = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=5.0,
            min_child_weight=5,
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_pipe = Pipeline([("scaler", StandardScaler()), ("xgb", xgb_core)])
        # XGBoost early stopping needs raw arrays — fit core after scaling
        X_tr_sc2 = StandardScaler().fit(X_tr).transform(X_tr)
        X_vl_sc2 = StandardScaler().fit(X_tr).transform(X_vl)
        xgb_core.fit(X_tr_sc2, y_tr, eval_set=[(X_vl_sc2, y_vl)], verbose=False)
        # Re-wrap in final pipeline (scaler fitted on X_tr)
        xgb_pipe = Pipeline([("scaler", StandardScaler().fit(X_tr)), ("xgb", xgb_core)])

        r2_vl_x  = r2_score(y_vl,  xgb_pipe.predict(X_vl))
        r2_te_x  = r2_score(y_te,  xgb_pipe.predict(X_te))
        rmse_te_x = np.sqrt(mean_squared_error(y_te, xgb_pipe.predict(X_te)))
        print(f"  [XGB] Val R²={r2_vl_x:.3f}  |  Test R²={r2_te_x:.3f}  RMSE={rmse_te_x:.3f}")

        joblib.dump(xgb_pipe, f"models/xgb_{tag}_{split_name}.pkl")
        results[f"XGB_{tag}_{split_name}"] = {
            "val_r2": r2_vl_x, "test_r2": r2_te_x, "test_rmse": rmse_te_x,
            "y_test": y_te, "y_pred": xgb_pipe.predict(X_te)
        }

        # ── SVR (Optuna tuning) ────────────────────────────────────────────
        print(f"  [SVR] Optuna search (60 trials)...")
        best_svr = tune_svr(X_tr_sc, y_tr, X_vl_sc, y_vl)
        svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", best_svr)])
        svr_pipe.fit(X_tr, y_tr)

        r2_vl_s  = r2_score(y_vl,  svr_pipe.predict(X_vl))
        r2_te_s  = r2_score(y_te,  svr_pipe.predict(X_te))
        rmse_te_s = np.sqrt(mean_squared_error(y_te, svr_pipe.predict(X_te)))
        print(f"  [SVR] Val R²={r2_vl_s:.3f}  |  Test R²={r2_te_s:.3f}  RMSE={rmse_te_s:.3f}")

        joblib.dump(svr_pipe, f"models/svr_{tag}_{split_name}.pkl")
        results[f"SVR_{tag}_{split_name}"] = {
            "val_r2": r2_vl_s, "test_r2": r2_te_s, "test_rmse": rmse_te_s,
            "y_test": y_te, "y_pred": svr_pipe.predict(X_te)
        }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("=== 5-fold CV 汇总（主要指标）===")
print(f"{'模型':<30} {'CV R² mean':>12} {'± std':>8}")
print("-" * 52)
for name, m in results.items():
    if "cv_r2_mean" in m:
        print(f"{name:<30} {m['cv_r2_mean']:>12.3f} {m['cv_r2_std']:>8.3f}")

print(f"\n=== 测试集汇总（scaffold / stratified / random）===")
print(f"{'模型':<40} {'Val R²':>8} {'Test R²':>8} {'Test RMSE':>10}")
print("-" * 70)
for name, m in results.items():
    if "val_r2" in m:
        print(f"{name:<40} {m['val_r2']:>8.3f} {m['test_r2']:>8.3f} {m['test_rmse']:>10.3f}")

joblib.dump(results, "models/train_results.pkl")
print("\n已保存训练结果到 models/train_results.pkl")
print("已保存所有模型到 models/ 目录")
