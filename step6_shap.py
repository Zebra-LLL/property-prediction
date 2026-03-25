"""
Step 6: SHAP Interpretability Analysis
Uses XGBoost + scaffold split models (best interpretability).
Generates:
  - SHAP summary plot (global feature importance)
  - SHAP force plot for top predicted compound

Inputs:  models/xgb_b2_scaffold.pkl
         models/xgb_b1_scaffold.pkl
         features/X_ecfp4.npy
         splits/scaffold_split.npz
         data_cleaned.csv

Outputs: results/shap_summary_b2.png
         results/shap_summary_b1.png
         results/shap_force_top_b2.png
         results/feature_importance_b2.csv
         results/feature_importance_b1.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df     = pd.read_csv("data_cleaned.csv")
X_fp   = np.load("features/X_ecfp4.npy")
sp     = np.load("splits/scaffold_split.npz")

test_idx = sp["test_idx"]

# ── SHAP analysis function ────────────────────────────────────────────────────
def shap_analysis(target, tag):
    model = joblib.load(f"models/xgb_{tag}_scaffold.pkl")

    mask = df[target].notna().values
    pos  = np.where(mask)[0]
    p2n  = {old: new for new, old in enumerate(pos)}

    te   = [i for i in test_idx if mask[i]]
    X_te = X_fp[[p2n[i] for i in te]]
    y_te = df.loc[mask, target].values[[p2n[i] for i in te]]

    print(f"\n[{target}] SHAP 分析，测试集样本数: {len(y_te)}")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    # ── Global summary plot ────────────────────────────────────────────────
    plt.figure()
    shap.summary_plot(shap_values, X_te, max_display=20,
                      feature_names=[f"bit_{i}" for i in range(X_te.shape[1])],
                      show=False)
    plt.title(f"SHAP Summary - {target}")
    plt.tight_layout()
    plt.savefig(f"results/shap_summary_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存 results/shap_summary_{tag}.png")

    # ── Feature importance table (mean |SHAP|) ─────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[::-1][:50]
    fi_df = pd.DataFrame({
        "feature":         [f"bit_{i}" for i in top_idx],
        "mean_abs_shap":   mean_abs_shap[top_idx].round(6),
        "ecfp4_bit_index": top_idx
    })
    fi_df.to_csv(f"results/feature_importance_{tag}.csv", index=False)
    print(f"已保存 results/feature_importance_{tag}.csv (top 50 bits)")

    # ── Force plot for top-predicted compound ──────────────────────────────
    top_i = int(np.argmax(model.predict(X_te)))
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[top_i],
        X_te[top_i],
        feature_names=[f"bit_{i}" for i in range(X_te.shape[1])],
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - Top Predicted {target}")
    plt.tight_layout()
    plt.savefig(f"results/shap_force_top_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存 results/shap_force_top_{tag}.png")

    return fi_df


# ── Run for B2 and B1 ─────────────────────────────────────────────────────────
fi_b2 = shap_analysis("pIC50_B2", "b2")
fi_b1 = shap_analysis("pIC50_B1", "b1")

print("\n[SHAP] Top 10 重要 ECFP4 bits (B2 模型):")
print(fi_b2.head(10).to_string(index=False))

print("\n[SHAP] Top 10 重要 ECFP4 bits (B1 模型):")
print(fi_b1.head(10).to_string(index=False))

print("\n注：ECFP4 bit 索引无法直接解读化学结构。")
print("    如需化学可解释性，请改用 RDKit 描述符特征（features/X_desc.npy）。")
