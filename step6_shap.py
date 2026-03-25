"""
Step 6: SHAP + Permutation Importance Interpretability Analysis
- XGBoost (from Pipeline): SHAP TreeExplainer
- SVR (from Pipeline): permutation importance
- Uses stratified_scaffold split (primary evaluation split)
- Uses X_combined features (ECFP4 + RDKit descriptors)

Inputs:  models/xgb_{b2,b1}_stratified_scaffold.pkl  (Pipeline)
         models/svr_{b2,b1}_stratified_scaffold.pkl  (Pipeline)
         features/X_combined.npy
         features/desc_names.txt
         splits/stratified_scaffold_split.npz
         data_cleaned.csv

Outputs: results/shap_summary_{b2,b1}.png
         results/shap_force_top_{b2,b1}.png
         results/feature_importance_{b2,b1}.csv   (SHAP, XGB)
         results/perm_importance_{b2,b1}.csv       (permutation, SVR)
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

os.makedirs("results", exist_ok=True)

# ── Load data and features ────────────────────────────────────────────────────
df      = pd.read_csv("data_cleaned.csv")
X_comb  = np.load("features/X_combined.npy")          # (415, 2189)
sp      = np.load("splits/stratified_scaffold_split.npz")
test_idx = sp["test_idx"]

# Feature names: 2048 ECFP4 bits + 141 RDKit descriptor names
ecfp_names = [f"bit_{i}" for i in range(2048)]
desc_names = open("features/desc_names.txt").read().splitlines()
feature_names = ecfp_names + desc_names  # length 2189


def patch_xgb_base_score(model):
    """Fix XGBoost >=2.0 base_score bracket-string format for SHAP compatibility."""
    try:
        booster = model.get_booster()
        raw = json.loads(booster.save_raw("json").decode())
        lmp = raw["learner"]["learner_model_param"]
        bs = lmp.get("base_score", "0.5")
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


def get_test_arrays(target, test_idx):
    """Return raw X_te and y_te for the test set (valid target only)."""
    mask = df[target].notna().values
    pos  = np.where(mask)[0]
    p2n  = {old: new for new, old in enumerate(pos)}
    te   = [i for i in test_idx if mask[i]]
    X_te = X_comb[[p2n[i] for i in te]]
    y_te = df.loc[mask, target].values[[p2n[i] for i in te]]
    return X_te, y_te


# ── XGBoost SHAP analysis ──────────────────────────────────────────────────────
def shap_analysis_xgb(target, tag):
    pipeline = joblib.load(f"models/xgb_{tag}_stratified_scaffold.pkl")
    scaler   = pipeline.named_steps["scaler"]
    xgb_core = patch_xgb_base_score(pipeline.named_steps["xgb"])

    X_te, y_te = get_test_arrays(target, test_idx)
    X_te_sc    = scaler.transform(X_te)

    print(f"\n[{target}] XGB SHAP 分析，测试集样本数: {len(y_te)}")
    print(f"  Test R² (pipeline): {r2_score(y_te, pipeline.predict(X_te)):.3f}")

    explainer   = shap.TreeExplainer(xgb_core)
    shap_values = explainer.shap_values(X_te_sc)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_te_sc, max_display=20,
                      feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary (XGB) - {target}")
    plt.tight_layout()
    plt.savefig(f"results/shap_summary_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存 results/shap_summary_{tag}.png")

    # Feature importance table
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[::-1][:50]
    fi_df = pd.DataFrame({
        "feature":       [feature_names[i] for i in top_idx],
        "mean_abs_shap": mean_abs_shap[top_idx].round(6),
        "feat_index":    top_idx
    })
    fi_df.to_csv(f"results/feature_importance_{tag}.csv", index=False)
    print(f"已保存 results/feature_importance_{tag}.csv (top 50 features)")

    # Force plot for top-predicted compound
    top_i = int(np.argmax(xgb_core.predict(X_te_sc)))
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[top_i],
        X_te_sc[top_i],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - Top Predicted {target}")
    plt.tight_layout()
    plt.savefig(f"results/shap_force_top_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存 results/shap_force_top_{tag}.png")

    return fi_df


# ── SVR Permutation Importance ─────────────────────────────────────────────────
def perm_importance_svr(target, tag, n_repeats=20):
    pipeline = joblib.load(f"models/svr_{tag}_stratified_scaffold.pkl")
    X_te, y_te = get_test_arrays(target, test_idx)

    print(f"\n[{target}] SVR 排列重要性，n_repeats={n_repeats}...")
    print(f"  Test R² (pipeline): {r2_score(y_te, pipeline.predict(X_te)):.3f}")

    result = permutation_importance(
        pipeline, X_te, y_te,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="r2"
    )

    top_idx = np.argsort(result.importances_mean)[::-1][:50]
    pi_df = pd.DataFrame({
        "feature":         [feature_names[i] for i in top_idx],
        "importance_mean": result.importances_mean[top_idx].round(6),
        "importance_std":  result.importances_std[top_idx].round(6),
        "feat_index":      top_idx
    })
    pi_df.to_csv(f"results/perm_importance_{tag}.csv", index=False)
    print(f"已保存 results/perm_importance_{tag}.csv (top 50 features)")
    return pi_df


# ── Run for B2 and B1 ─────────────────────────────────────────────────────────
fi_b2 = shap_analysis_xgb("pIC50_B2", "b2")
fi_b1 = shap_analysis_xgb("pIC50_B1", "b1")

pi_b2 = perm_importance_svr("pIC50_B2", "b2")
pi_b1 = perm_importance_svr("pIC50_B1", "b1")

print("\n[SHAP XGB] Top 10 重要特征 (B2):")
print(fi_b2.head(10).to_string(index=False))

print("\n[SHAP XGB] Top 10 重要特征 (B1):")
print(fi_b1.head(10).to_string(index=False))

print("\n[SVR 排列重要性] Top 10 重要特征 (B2):")
print(pi_b2.head(10).to_string(index=False))

print("\n注：ECFP4 bit 索引无法直接解读化学结构。")
print("    RDKit 描述符特征（bit_2048+）具备直接化学可解释性。")
