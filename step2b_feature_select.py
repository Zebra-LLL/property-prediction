"""
Step 2b: Feature Selection
Three-stage pipeline on X_combined (415, 2189):
  1. Low-variance filter  (threshold = 0.01)
  2. Correlation filter   (drop one from each pair with |r| > 0.95)
  3. RF importance        (average over B2 + B1; keep top 300)

Also runs stratified-scaffold 5-fold CV before/after to compare R².

Inputs:  features/X_combined.npy
         data_cleaned.csv  (requires 'scaffold' column)

Outputs: features/X_selected.npy          (415, ≤300)
         features/selected_feature_idx.npy (indices into X_combined)
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score
import os

warnings.filterwarnings("ignore")
os.makedirs("features", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df  = pd.read_csv("data_cleaned.csv")
X   = np.load("features/X_combined.npy")   # (415, 2189)
n_orig = X.shape[1]
print(f"Original features: {n_orig}")

# ── Stage 1: Variance filter ──────────────────────────────────────────────────
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
mask_vt = vt.get_support()
print(f"After variance filter (threshold=0.01): {X_vt.shape[1]} features "
      f"(removed {n_orig - X_vt.shape[1]})")

# ── Stage 2: Correlation filter (|r| > 0.95) ─────────────────────────────────
corr = np.corrcoef(X_vt.T)
n = X_vt.shape[1]

# Sort features by variance descending (keep higher-variance feature in a pair)
variances = X_vt.var(axis=0)
sorted_idx = np.argsort(variances)[::-1]

keep = np.ones(n, dtype=bool)
for i_pos, i in enumerate(sorted_idx):
    if not keep[i]:
        continue
    for j in sorted_idx[i_pos + 1:]:
        if keep[j] and abs(corr[i, j]) > 0.95:
            keep[j] = False

mask_corr = keep
X_corr = X_vt[:, mask_corr]
print(f"After correlation filter (|r|>0.95):   {X_corr.shape[1]} features "
      f"(removed {mask_vt.sum() - mask_corr.sum()})")

# ── Stage 3: RF importance — average over B2 and B1 ──────────────────────────
importance_sum = np.zeros(X_corr.shape[1])
n_targets_used = 0

for target in ["pIC50_B2", "pIC50_B1"]:
    y_series = df[target]
    mask_y   = y_series.notna().values
    y_valid  = y_series.dropna().values
    X_valid  = X_corr[mask_y]

    rf = RandomForestRegressor(n_estimators=500, max_features=0.3,
                               min_samples_leaf=3, n_jobs=-1, random_state=42)
    sc = StandardScaler().fit(X_valid)
    rf.fit(sc.transform(X_valid), y_valid)
    importance_sum += rf.feature_importances_
    n_targets_used += 1

avg_importance = importance_sum / n_targets_used

TOP_K = 300
top_idx_local = np.argsort(avg_importance)[::-1][:TOP_K]

# Map back to original X_combined indices
idx_after_vt   = np.where(mask_vt)[0]          # indices in X_combined after vt
idx_after_corr = idx_after_vt[mask_corr]        # indices in X_combined after corr
selected_orig_idx = idx_after_corr[top_idx_local]  # final indices in X_combined

X_selected = X[:, selected_orig_idx]
print(f"After RF importance (top {TOP_K}):       {X_selected.shape[1]} features")

np.save("features/X_selected.npy",          X_selected)
np.save("features/selected_feature_idx.npy", selected_orig_idx)
print(f"\nSaved features/X_selected.npy  shape={X_selected.shape}")
print(f"Saved features/selected_feature_idx.npy  ({len(selected_orig_idx)} indices)")

# ── Stratified-scaffold CV splitter ───────────────────────────────────────────
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


def cv_r2_splits(pipeline_factory, X_all, y_all, cv_splits):
    scores = []
    for tr, te in cv_splits:
        p = pipeline_factory()
        p.fit(X_all[tr], y_all[tr])
        scores.append(r2_score(y_all[te], p.predict(X_all[te])))
    return np.array(scores)


# ── Before / After CV comparison ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("Stratified-scaffold 5-fold CV: Before vs After feature selection")
print(f"{'='*70}")

feature_sets = {
    f"X_combined ({n_orig}d)": X,
    f"X_selected ({TOP_K}d)":  X_selected,
}

def make_rf():
    return Pipeline([("sc", StandardScaler()),
                     ("rf", RandomForestRegressor(n_estimators=500, max_features=0.3,
                                                   min_samples_leaf=3, n_jobs=-1, random_state=42))])
def make_xgb():
    return Pipeline([("sc", StandardScaler()),
                     ("xgb", xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                               max_depth=4, subsample=0.8, colsample_bytree=0.6,
                                               reg_alpha=0.5, reg_lambda=5.0, min_child_weight=5,
                                               verbosity=0, n_jobs=-1, random_state=42))])
def make_svr():
    return Pipeline([("sc", StandardScaler()),
                     ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"))])

model_factories = {"RF": make_rf, "XGB": make_xgb, "SVR": make_svr}

all_results = {}   # (target, feat_label, model) → (mean, std)

for target, tag in [("pIC50_B2", "B2"), ("pIC50_B1", "B1")]:
    y_series = df[target]
    mask_y   = y_series.notna().values
    valid_orig_idx = np.where(mask_y)[0]
    y_all    = y_series.dropna().values

    cv_splits = stratified_scaffold_cv(valid_orig_idx, df["scaffold"])

    for feat_label, X_feat in feature_sets.items():
        X_valid = X_feat[mask_y]
        for model_name, factory in model_factories.items():
            scores = cv_r2_splits(factory, X_valid, y_all, cv_splits)
            all_results[(tag, feat_label, model_name)] = (scores.mean(), scores.std())

# ── Print comparison table ─────────────────────────────────────────────────────
print(f"\n{'Target':<5} {'Model':<5} {'X_combined (2189d)':>20}   {'X_selected (300d)':>20}   {'Δ R²':>7}")
print("-" * 65)
for target, tag in [("pIC50_B2", "B2"), ("pIC50_B1", "B1")]:
    for model_name in ["RF", "XGB", "SVR"]:
        feat_labels = list(feature_sets.keys())
        m_before, s_before = all_results[(tag, feat_labels[0], model_name)]
        m_after,  s_after  = all_results[(tag, feat_labels[1], model_name)]
        delta = m_after - m_before
        sign  = "+" if delta >= 0 else ""
        print(f"{tag:<5} {model_name:<5} "
              f"{m_before:>8.3f} ± {s_before:.3f}      "
              f"{m_after:>8.3f} ± {s_after:.3f}   "
              f"{sign}{delta:>+.3f}")
    print()

# ── Bar chart: Before vs After per model/target ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
targets_info = [("B2", "CYP11B2"), ("B1", "CYP11B1")]
model_names  = ["RF", "XGB", "SVR"]
x = np.arange(len(model_names))
width = 0.35
colors = {"before": "#7CB9E8", "after": "#003F87"}
feat_labels = list(feature_sets.keys())

for ax_i, (tag, label) in enumerate(targets_info):
    ax = axes[ax_i]
    vals_before = [all_results[(tag, feat_labels[0], m)][0] for m in model_names]
    errs_before = [all_results[(tag, feat_labels[0], m)][1] for m in model_names]
    vals_after  = [all_results[(tag, feat_labels[1], m)][0] for m in model_names]
    errs_after  = [all_results[(tag, feat_labels[1], m)][1] for m in model_names]

    b1 = ax.bar(x - width/2, vals_before, width, label=f"X_combined (2189d)",
                color=colors["before"], alpha=0.85, edgecolor="k", lw=0.5,
                yerr=errs_before, capsize=3)
    b2 = ax.bar(x + width/2, vals_after,  width, label=f"X_selected (300d)",
                color=colors["after"],  alpha=0.85, edgecolor="k", lw=0.5,
                yerr=errs_after,  capsize=3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Scaffold-CV R²", fontsize=11)
    ax.set_title(f"{label}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(max(vals_before), max(vals_after)) * 1.20 + 0.05)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

fig.suptitle("Stratified-Scaffold 5-Fold CV — Feature Selection Comparison",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/feature_selection_comparison.png", dpi=150, bbox_inches="tight")
print("Saved → figures/feature_selection_comparison.png")
