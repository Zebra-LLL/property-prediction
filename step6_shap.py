"""
Step 6: SHAP + Permutation Importance Interpretability Analysis

Models:
  - XGBoost final models → SHAP TreeExplainer (global summary + force plot)
  - SVR final models     → Permutation importance

Data: all valid samples (no test split — final models trained on full data)
Features: X_selected (300d), feature names reconstructed from selected_feature_idx

Inputs:  models/xgb_{b2,b1}_final.pkl
         models/final_model_B2.pkl  (SVR)
         models/final_model_B1.pkl  (SVR)
         features/X_selected.npy
         features/selected_feature_idx.npy
         features/desc_names.txt    (141 filtered RDKit descriptor names)
         data_cleaned.csv

Outputs: results/shap_summary_{b2,b1}.png
         results/shap_force_top_{b2,b1}.png
         results/feature_importance_xgb_{b2,b1}.csv
         results/perm_importance_svr_{b2,b1}.csv
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from utils import IndexSelector   # required for feature_selector.pkl

os.makedirs("results", exist_ok=True)

# ── Load data and features ────────────────────────────────────────────────────
df         = pd.read_csv("data_cleaned.csv")
X_selected = np.load("features/X_selected.npy")          # (415, 300)
sel_idx    = np.load("features/selected_feature_idx.npy") # (300,) — indices into X_combined

# ── Feature names for the 300 selected features ───────────────────────────────
# X_combined = [bit_0 … bit_2047] + [desc_names (141 filtered)]  → 2189 total
ecfp_names_all = [f"ECFP4_bit{i}" for i in range(2048)]
desc_names_filtered = open("features/desc_names.txt").read().splitlines()  # 141 names
feature_names_combined = ecfp_names_all + desc_names_filtered              # 2189 names
feature_names_selected = [feature_names_combined[i] for i in sel_idx]     # 300 names

print(f"Selected feature names: {len(feature_names_selected)} total")
n_ecfp = sum(1 for n in feature_names_selected if n.startswith("ECFP4"))
n_desc = sum(1 for n in feature_names_selected if not n.startswith("ECFP4"))
print(f"  ECFP4 bits: {n_ecfp},  RDKit descriptors: {n_desc}")


# ── Helper: get valid (X, y) for a target ─────────────────────────────────────
def get_valid(target):
    mask = df[target].notna().values
    return X_selected[mask], df[target].dropna().values


# ── XGBoost SHAP patch ────────────────────────────────────────────────────────
def patch_xgb_base_score(model):
    """Fix XGBoost >=2.0 base_score bracket-string format for SHAP."""
    try:
        booster = model.get_booster()
        raw = json.loads(booster.save_raw("json").decode())
        lmp = raw["learner"]["learner_model_param"]
        bs  = lmp.get("base_score", "0.5")
        if isinstance(bs, str) and bs.startswith("["):
            lmp["base_score"] = str(float(bs.strip("[]")))
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                json.dump(raw, f)
                tmp_path = f.name
            booster.load_model(tmp_path)
            os.unlink(tmp_path)
    except Exception:
        pass
    return model


# ── XGBoost SHAP analysis ─────────────────────────────────────────────────────
def shap_analysis_xgb(target, tag):
    pipeline = joblib.load(f"models/xgb_{tag}_final.pkl")
    scaler   = pipeline.named_steps["sc"]
    xgb_core = patch_xgb_base_score(pipeline.named_steps["xgb"])

    X_all, y_all = get_valid(target)
    X_sc = scaler.transform(X_all)

    print(f"\n[{target}] XGB SHAP  n={len(y_all)}")
    print(f"  Train R² (in-sample): {r2_score(y_all, pipeline.predict(X_all)):.3f}")

    explainer   = shap.TreeExplainer(xgb_core)
    shap_values = explainer.shap_values(X_sc)   # (n, 300)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sc, max_display=20,
                      feature_names=feature_names_selected, show=False)
    plt.title(f"SHAP Summary — XGBoost {target}")
    plt.tight_layout()
    plt.savefig(f"results/shap_summary_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved results/shap_summary_{tag}.png")

    # Feature importance CSV (top 50)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:50]
    fi_df = pd.DataFrame({
        "feature":       [feature_names_selected[i] for i in top_idx],
        "mean_abs_shap": mean_abs_shap[top_idx].round(6),
        "feat_index_in_selected": top_idx,
        "feat_index_in_combined": sel_idx[top_idx],
    })
    fi_df.to_csv(f"results/feature_importance_xgb_{tag}.csv", index=False)
    print(f"  Saved results/feature_importance_xgb_{tag}.csv (top 50)")

    # Force plot for the compound with highest predicted pIC50
    top_i = int(np.argmax(xgb_core.predict(X_sc)))
    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[top_i], X_sc[top_i],
        feature_names=feature_names_selected,
        matplotlib=True, show=False
    )
    plt.title(f"SHAP Force Plot — Top Predicted {target}")
    plt.tight_layout()
    plt.savefig(f"results/shap_force_top_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved results/shap_force_top_{tag}.png")

    return fi_df


# ── SVR Permutation Importance ────────────────────────────────────────────────
def perm_importance_svr(target, tag, n_repeats=20):
    model_path = f"models/final_model_B{tag.upper()[-1]}.pkl"
    pipeline   = joblib.load(model_path)

    X_all, y_all = get_valid(target)

    print(f"\n[{target}] SVR permutation importance  n={len(y_all)}, repeats={n_repeats}")
    print(f"  Train R² (in-sample): {r2_score(y_all, pipeline.predict(X_all)):.3f}")

    result = permutation_importance(
        pipeline, X_all, y_all,
        n_repeats=n_repeats, random_state=42, n_jobs=-1, scoring="r2"
    )

    top_idx = np.argsort(result.importances_mean)[::-1][:50]
    pi_df = pd.DataFrame({
        "feature":         [feature_names_selected[i] for i in top_idx],
        "importance_mean": result.importances_mean[top_idx].round(6),
        "importance_std":  result.importances_std[top_idx].round(6),
        "feat_index_in_selected": top_idx,
        "feat_index_in_combined": sel_idx[top_idx],
    })
    pi_df.to_csv(f"results/perm_importance_svr_{tag}.csv", index=False)
    print(f"  Saved results/perm_importance_svr_{tag}.csv (top 50)")
    return pi_df


# ── Run ───────────────────────────────────────────────────────────────────────
fi_b2 = shap_analysis_xgb("pIC50_B2", "b2")
fi_b1 = shap_analysis_xgb("pIC50_B1", "b1")

pi_b2 = perm_importance_svr("pIC50_B2", "b2")
pi_b1 = perm_importance_svr("pIC50_B1", "b1")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("[XGB SHAP] Top 10 features — CYP11B2:")
print(fi_b2[["feature", "mean_abs_shap"]].head(10).to_string(index=False))

print("\n[XGB SHAP] Top 10 features — CYP11B1:")
print(fi_b1[["feature", "mean_abs_shap"]].head(10).to_string(index=False))

print("\n[SVR Permutation] Top 10 features — CYP11B2:")
print(pi_b2[["feature", "importance_mean"]].head(10).to_string(index=False))

print("\n[SVR Permutation] Top 10 features — CYP11B1:")
print(pi_b1[["feature", "importance_mean"]].head(10).to_string(index=False))

print("\n注：ECFP4_bit* 特征无直接化学解读；RDKit 描述符名称即化学含义。")
