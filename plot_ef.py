"""
Enrichment Factor (EF@10%, EF@20%) via 5-fold CV for XGBoost and SVR.

Active definition:
  CYP11B2: pIC50_B2 > 7.5   (high potency)
  CYP11B1: pIC50_B1 < 6.0   (low off-target, sorted ascending by prediction)

Ranking:  B2 → descending predicted pIC50
          B1 → ascending  predicted pIC50

EF@x% = (actives in top x%) / (x% × total_actives)

Outputs: figures/ef_barplot.png
         prints EF table
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_cleaned.csv")
X  = np.load("features/X_combined.npy")

TARGETS = [
    ("pIC50_B2", "CYP11B2", lambda y: y > 7.5, "descending"),
    ("pIC50_B1", "CYP11B1", lambda y: y < 6.0, "ascending"),
]

MODELS = {
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

def cv_collect(model_factory, X_all, y_all, n_splits=5, seed=42):
    """Return (y_true, y_pred) concatenated across all 5 folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tr, te in kf.split(X_all):
        m = model_factory()
        m.fit(X_all[tr], y_all[tr])
        yt.append(y_all[te])
        yp.append(m.predict(X_all[te]))
    return np.concatenate(yt), np.concatenate(yp)

def enrichment_factor(y_true, y_pred, is_active_fn, direction, frac):
    """
    EF@frac for a virtual screening scenario.
    direction: 'descending' → rank highest pred first
               'ascending'  → rank lowest pred first
    """
    n = len(y_true)
    n_top = max(1, int(np.ceil(frac * n)))
    actives = is_active_fn(y_true)
    n_actives_total = actives.sum()
    if n_actives_total == 0:
        return np.nan

    if direction == "descending":
        ranked_idx = np.argsort(y_pred)[::-1]
    else:
        ranked_idx = np.argsort(y_pred)

    top_idx = ranked_idx[:n_top]
    n_actives_top = actives[top_idx].sum()

    ef = (n_actives_top / n_top) / (n_actives_total / n)
    return ef

# ── Compute ────────────────────────────────────────────────────────────────────
records = []
for model_name, factory in MODELS.items():
    for target, label, is_active_fn, direction in TARGETS:
        y_series = df[target]
        mask = y_series.notna().values
        y_all = y_series.dropna().values
        X_all = X[mask]

        n_actives = is_active_fn(y_all).sum()
        active_rate = n_actives / len(y_all)

        yt, yp = cv_collect(factory, X_all, y_all)
        r2 = r2_score(yt, yp)

        ef10 = enrichment_factor(yt, yp, is_active_fn, direction, 0.10)
        ef20 = enrichment_factor(yt, yp, is_active_fn, direction, 0.20)

        records.append({
            "Model":   model_name,
            "Target":  label,
            "n":       len(y_all),
            "Actives": n_actives,
            "Active%": f"{active_rate*100:.1f}%",
            "R²":      r2,
            "EF@10%":  ef10,
            "EF@20%":  ef20,
        })
        print(f"{model_name:8s} {label:10s}  n={len(y_all):3d}  "
              f"actives={n_actives:3d} ({active_rate*100:.0f}%)  "
              f"R²={r2:.3f}  EF@10%={ef10:.2f}  EF@20%={ef20:.2f}")

df_res = pd.DataFrame(records)

# ── Print table ────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<10} {'Target':<10} {'n':>4} {'Actives':>7} {'Active%':>8} "
      f"{'R²':>6} {'EF@10%':>8} {'EF@20%':>8}")
print("-"*65)
for _, r in df_res.iterrows():
    print(f"{r['Model']:<10} {r['Target']:<10} {r['n']:>4} {r['Actives']:>7} "
          f"{r['Active%']:>8} {r['R²']:>6.3f} {r['EF@10%']:>8.2f} {r['EF@20%']:>8.2f}")

# random baseline EF = 1.0
print(f"\n  Random baseline: EF@10% = 1.00, EF@20% = 1.00")

# ── Bar plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fracs  = ["EF@10%", "EF@20%"]
colors = {"XGBoost": "#4C72B0", "SVR": "#DD8452"}
x = np.arange(2)   # B2, B1
width = 0.35

for ax_idx, frac in enumerate(fracs):
    ax = axes[ax_idx]
    for i, (model_name, _) in enumerate(MODELS.items()):
        vals = [df_res[(df_res.Model == model_name) & (df_res.Target == lbl)][frac].values[0]
                for _, lbl, _, _ in TARGETS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model_name,
                      color=colors[model_name], alpha=0.85, edgecolor="k", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # random baseline
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Random (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(["CYP11B2\n(pIC50 > 7.5)", "CYP11B1\n(pIC50 < 6.0)"], fontsize=11)
    ax.set_ylabel("Enrichment Factor", fontsize=11)
    ax.set_title(frac, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(df_res[frac].max() * 1.25, 2.5))
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("5-Fold CV Enrichment Factor (EF@10% and EF@20%)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/ef_barplot.png", dpi=150, bbox_inches="tight")
print("\nSaved → figures/ef_barplot.png")
