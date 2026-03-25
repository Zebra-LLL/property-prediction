"""
Stratified-scaffold 5-fold CV parity plots for XGBoost and SVR (B2 and B1).

Outputs: figures/cv_parity_xgb_svr.png
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_cleaned.csv")
X  = np.load("features/X_selected.npy")   # (415, 300) — after feature selection

# ── Stratified-scaffold CV (shared with step4_train.py) ───────────────────────
def stratified_scaffold_cv(valid_orig_idx, scaffold_col, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    valid_orig_idx = np.asarray(valid_orig_idx)
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


TARGETS = [("pIC50_B2", "CYP11B2"), ("pIC50_B1", "CYP11B1")]
MODELS  = {
    "XGBoost": lambda: Pipeline([
        ("sc", StandardScaler()),
        ("m",  xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=5.0, min_child_weight=5,
            verbosity=0, n_jobs=-1, random_state=42))
    ]),
    "SVR": lambda: Pipeline([
        ("sc", StandardScaler()),
        ("m",  SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"))
    ]),
}


def cv_parity(model_factory, X_all, y_all, cv_splits):
    """Return (y_true, y_pred, fold_ids) from stratified-scaffold CV."""
    yt, yp, folds = [], [], []
    for fold, (tr, te) in enumerate(cv_splits):
        m = model_factory()
        m.fit(X_all[tr], y_all[tr])
        yt.append(y_all[te])
        yp.append(m.predict(X_all[te]))
        folds.append(np.full(len(te), fold))
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(folds)


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
FOLD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

for col, (model_name, factory) in enumerate(MODELS.items()):
    for row, (target, label) in enumerate(TARGETS):
        ax = axes[row][col]

        y_series = df[target]
        mask = y_series.notna().values
        y_all = y_series.dropna().values
        X_all = X[mask]
        valid_orig_idx = np.where(mask)[0]

        cv_splits = stratified_scaffold_cv(valid_orig_idx, df["scaffold"])
        yt, yp, folds = cv_parity(factory, X_all, y_all, cv_splits)
        r2 = r2_score(yt, yp)

        for f in range(5):
            idx = folds == f
            ax.scatter(yt[idx], yp[idx],
                       color=FOLD_COLORS[f], alpha=0.65, s=22,
                       label=f"Fold {f+1}" if row == 0 and col == 0 else "")

        lo = min(yt.min(), yp.min()) - 0.2
        hi = max(yt.max(), yp.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Experimental pIC50", fontsize=11)
        ax.set_ylabel("Predicted pIC50",    fontsize=11)
        ax.set_title(f"{model_name} — {label}", fontsize=12, fontweight="bold")
        ax.text(0.05, 0.93, f"$R^2$ = {r2:.3f}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=FOLD_COLORS[f], markersize=8,
                      label=f"Fold {f+1}") for f in range(5)]
fig.legend(handles=handles, loc="lower center", ncol=5,
           frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Stratified-Scaffold 5-Fold CV Parity Plots",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/cv_parity_xgb_svr.png", dpi=150, bbox_inches="tight")
print("Saved → figures/cv_parity_xgb_svr.png")
