"""
Step 4: Model Training

Primary evaluation:  Stratified-scaffold 5-fold CV
  - Each scaffold group's compounds distributed proportionally (round-robin)
    across all 5 folds → every fold sees the full structural diversity
  - More realistic than random CV; avoids the extreme extrapolation of
    whole-scaffold GroupKFold

Final models:  Trained on ALL valid data (CV-selected hyperparams)
  - No fixed test set; CV R² is the reported performance metric

Inputs:  data_cleaned.csv  (requires 'scaffold' column from step3)
         features/X_combined.npy  (415, 2189) — ECFP4 + RDKit descriptors

Outputs: models/{rf,xgb,svr}_{b2,b1}_final.pkl
         models/train_results.pkl
"""

import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_cleaned.csv")
X  = np.load("features/X_combined.npy")   # (415, 2189)

# ── Stratified-scaffold CV splitter ───────────────────────────────────────────
def stratified_scaffold_cv(valid_orig_idx, scaffold_col, n_splits=5, seed=42):
    """
    Distribute each scaffold group's compounds round-robin across n_splits folds.
    Each fold therefore contains ~1/n of every scaffold — structurally
    representative of the full dataset.

    Parameters
    ----------
    valid_orig_idx : array-like, shape (n_valid,)
        Row indices into the full df for samples with non-NaN target.
    scaffold_col : pd.Series
        df['scaffold'] (indexed by original row position).

    Returns
    -------
    list of (train_local_idx, test_local_idx) — indices into the valid array.
    """
    rng = np.random.default_rng(seed)
    valid_orig_idx = np.asarray(valid_orig_idx)

    # Group local positions by scaffold
    groups = defaultdict(list)
    for local_i, orig_i in enumerate(valid_orig_idx):
        groups[scaffold_col.iloc[orig_i]].append(local_i)

    fold_assign = np.empty(len(valid_orig_idx), dtype=int)
    for members in groups.values():
        perm = rng.permutation(members)
        for j, idx in enumerate(perm):
            fold_assign[idx] = j % n_splits

    return [(np.where(fold_assign != f)[0], np.where(fold_assign == f)[0])
            for f in range(n_splits)]


# ── Helpers ────────────────────────────────────────────────────────────────────
def cv_r2(pipeline_factory, X_all, y_all, cv_splits):
    """Return per-fold R² array."""
    scores = []
    for tr, te in cv_splits:
        p = pipeline_factory()
        p.fit(X_all[tr], y_all[tr])
        scores.append(r2_score(y_all[te], p.predict(X_all[te])))
    return np.array(scores)


def tune_svr(X_sc, y, cv_splits, n_trials=60):
    """
    Tune SVR hyperparams by maximising mean scaffold-CV R².
    X_sc must already be StandardScaler-transformed.
    Returns the best unfitted SVR.
    """
    def objective(trial):
        C   = trial.suggest_float("C",       1e-1, 1e3, log=True)
        eps = trial.suggest_float("epsilon", 0.01, 0.5)
        g   = trial.suggest_float("gamma",   1e-4, 1e-1, log=True)
        scores = []
        for tr, te in cv_splits:
            m = SVR(kernel="rbf", C=C, epsilon=eps, gamma=g)
            m.fit(X_sc[tr], y[tr])
            scores.append(r2_score(y[te], m.predict(X_sc[te])))
        return -np.mean(scores)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"    Best SVR: C={bp['C']:.3g}, ε={bp['epsilon']:.3g}, "
          f"γ={bp['gamma']:.3g}")
    return SVR(kernel="rbf", C=bp["C"], epsilon=bp["epsilon"], gamma=bp["gamma"])


# ── Main loop ─────────────────────────────────────────────────────────────────
results = {}

for target in ["pIC50_B2", "pIC50_B1"]:
    tag = "b2" if "B2" in target else "b1"
    y_series = df[target]

    mask = y_series.notna().values
    valid_orig_idx = np.where(mask)[0]
    y_all = y_series.dropna().values
    X_all = X[mask]
    n = len(y_all)

    print(f"\n{'='*65}")
    print(f"Target: {target}  n={n}")

    cv_splits = stratified_scaffold_cv(valid_orig_idx, df["scaffold"], n_splits=5)
    fold_sizes = [len(te) for _, te in cv_splits]
    print(f"  Fold sizes (test): {fold_sizes}")

    # ── CV evaluation ──────────────────────────────────────────────────────
    cv_rf = cv_r2(
        lambda: Pipeline([
            ("sc", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=500, max_features=0.3,
                                          min_samples_leaf=3, n_jobs=-1, random_state=42))
        ]), X_all, y_all, cv_splits)

    cv_xgb = cv_r2(
        lambda: Pipeline([
            ("sc", StandardScaler()),
            ("xgb", xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                      max_depth=4, subsample=0.8, colsample_bytree=0.6,
                                      reg_alpha=0.5, reg_lambda=5.0, min_child_weight=5,
                                      verbosity=0, n_jobs=-1, random_state=42))
        ]), X_all, y_all, cv_splits)

    cv_svr = cv_r2(
        lambda: Pipeline([
            ("sc", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"))
        ]), X_all, y_all, cv_splits)

    print(f"  RF  scaffold-CV R²: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}  "
          f"  folds: {[round(v,3) for v in cv_rf]}")
    print(f"  XGB scaffold-CV R²: {cv_xgb.mean():.3f} ± {cv_xgb.std():.3f}  "
          f"  folds: {[round(v,3) for v in cv_xgb]}")
    print(f"  SVR scaffold-CV R²: {cv_svr.mean():.3f} ± {cv_svr.std():.3f}  "
          f"  folds: {[round(v,3) for v in cv_svr]}")

    results[f"RF_{tag}_cv"]  = {"cv_r2_mean": cv_rf.mean(),
                                 "cv_r2_std":  cv_rf.std(),
                                 "cv_r2_folds": cv_rf.tolist()}
    results[f"XGB_{tag}_cv"] = {"cv_r2_mean": cv_xgb.mean(),
                                 "cv_r2_std":  cv_xgb.std(),
                                 "cv_r2_folds": cv_xgb.tolist()}
    results[f"SVR_{tag}_cv"] = {"cv_r2_mean": cv_svr.mean(),
                                 "cv_r2_std":  cv_svr.std(),
                                 "cv_r2_folds": cv_svr.tolist()}

    # ── Tune SVR for the final model (Optuna over scaffold-CV) ────────────
    print(f"  [SVR] Optuna tuning (60 trials)...")
    scaler_all = StandardScaler().fit(X_all)
    X_all_sc   = scaler_all.transform(X_all)
    best_svr   = tune_svr(X_all_sc, y_all, cv_splits, n_trials=60)

    # ── Final models: ALL valid data ───────────────────────────────────────
    rf_final = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=500, max_features=0.3,
                                      min_samples_leaf=3, n_jobs=-1, random_state=42))
    ])
    rf_final.fit(X_all, y_all)

    xgb_final = Pipeline([
        ("sc", StandardScaler()),
        ("xgb", xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=4, subsample=0.8, colsample_bytree=0.6,
                                   reg_alpha=0.5, reg_lambda=5.0, min_child_weight=5,
                                   verbosity=0, n_jobs=-1, random_state=42))
    ])
    xgb_final.fit(X_all, y_all)

    svr_final = Pipeline([("sc", StandardScaler()), ("svr", best_svr)])
    svr_final.fit(X_all, y_all)

    joblib.dump(rf_final,  f"models/rf_{tag}_final.pkl")
    joblib.dump(xgb_final, f"models/xgb_{tag}_final.pkl")
    joblib.dump(svr_final, f"models/svr_{tag}_final.pkl")
    print(f"  Saved final models (trained on n={n}): "
          f"rf/xgb/svr_{tag}_final.pkl")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("=== Stratified-Scaffold 5-fold CV 汇总 ===")
print(f"{'模型':<22} {'mean R²':>9} {'± std':>7}  per-fold R²")
print("-" * 65)
for name, m in results.items():
    folds_str = "  ".join(f"{v:.3f}" for v in m["cv_r2_folds"])
    print(f"{name:<22} {m['cv_r2_mean']:>9.3f} {m['cv_r2_std']:>7.3f}  [{folds_str}]")

joblib.dump(results, "models/train_results.pkl")
print("\n已保存训练结果 → models/train_results.pkl")
print("已保存最终模型 → models/{rf,xgb,svr}_{b2,b1}_final.pkl")
