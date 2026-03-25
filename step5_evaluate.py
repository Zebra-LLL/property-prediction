"""
Step 5: Model Evaluation
- Compute R², RMSE, MAE, Pearson r for all models
- Report 5-fold CV R² (primary metric) and held-out test R² (secondary)
- Generate parity plots
- Enrichment factor analysis
- Comparison summary table

Inputs:  models/train_results.pkl  (from step4_train.py)
Outputs: results/metrics_summary.csv
         results/cv_summary.csv
         results/parity_*.png
         results/enrichment_summary.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

os.makedirs("results", exist_ok=True)

results = joblib.load("models/train_results.pkl")

# ── Evaluation functions ──────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    r, _ = stats.pearsonr(y_true, y_pred)
    if label:
        print(f"[{label}] R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f} | Pearson r={r:.3f}")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": r}


def plot_parity(y_true, y_pred, title="Parity Plot", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue")

    lo = min(float(y_true.min()), float(y_pred.min())) - 0.3
    hi = max(float(y_true.max()), float(y_pred.max())) + 0.3
    lims = [lo, hi]

    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.plot(lims, [l + 0.5 for l in lims], "r--", linewidth=0.8,
            alpha=0.5, label="±0.5 pIC50")
    ax.plot(lims, [l - 0.5 for l in lims], "r--", linewidth=0.8, alpha=0.5)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.93, f"R²={r2:.3f}\nRMSE={rmse:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Experimental pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存图表: {save_path}")
    plt.close()


def enrichment_factor(y_true, y_pred, top_frac=0.1, activity_cutoff=7.5):
    n_total = len(y_true)
    n_top   = max(1, int(n_total * top_frac))

    top_idx      = np.argsort(y_pred)[::-1][:n_top]
    hits_in_top  = int(np.sum(y_true[top_idx] > activity_cutoff))
    hits_total   = int(np.sum(y_true > activity_cutoff))

    random_rate = hits_total / n_total
    top_rate    = hits_in_top / n_top
    ef = top_rate / random_rate if random_rate > 0 else 0

    print(f"  EF@{int(top_frac*100)}%: {ef:.2f}x  "
          f"(top {n_top} 中命中 {hits_in_top}/{n_top}, "
          f"总命中率 {random_rate*100:.1f}%)")
    return ef


# ── CV summary (primary metric) ───────────────────────────────────────────────
cv_rows = []
print("=== 5-fold CV 汇总（主要指标）===")
for model_name, m in results.items():
    if "cv_r2_mean" in m:
        print(f"  {model_name:<30} R²={m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f}")
        cv_rows.append({"model": model_name,
                        "cv_r2_mean": round(m["cv_r2_mean"], 4),
                        "cv_r2_std": round(m["cv_r2_std"], 4)})

if cv_rows:
    pd.DataFrame(cv_rows).to_csv("results/cv_summary.csv", index=False)
    print("已保存 results/cv_summary.csv")

# ── Evaluate all held-out test models ─────────────────────────────────────────
metrics_rows = []
ef_rows      = []

for model_name, m in results.items():
    if "y_test" not in m:   # skip CV-only entries
        continue
    y_true = np.array(m["y_test"])
    y_pred = np.array(m["y_pred"])

    print(f"\n{model_name}")
    met = evaluate_model(y_true, y_pred, label=model_name)
    metrics_rows.append({"model": model_name, **met})

    # Parity plot
    safe_name = model_name.lower().replace(" ", "_")
    plot_parity(y_true, y_pred,
                title=model_name,
                save_path=f"results/parity_{safe_name}.png")

    # Enrichment factor (for B2 models only, cutoff = 7.5 pIC50)
    if "b2" in model_name.lower():
        print(f"  富集分析 ({model_name}):")
        for frac in [0.05, 0.10, 0.20]:
            ef = enrichment_factor(y_true, y_pred,
                                   top_frac=frac, activity_cutoff=7.5)
            ef_rows.append({
                "model": model_name,
                "top_frac": frac,
                "EF": round(ef, 2)
            })

# ── Save summary tables ───────────────────────────────────────────────────────
metrics_df = pd.DataFrame(metrics_rows).set_index("model")
metrics_df = metrics_df.round(4)
metrics_df.to_csv("results/metrics_summary.csv")
print(f"\n{'='*70}")
print("模型指标汇总:")
print(metrics_df.to_string())
print(f"\n已保存 results/metrics_summary.csv")

if ef_rows:
    ef_df = pd.DataFrame(ef_rows)
    ef_df.to_csv("results/enrichment_summary.csv", index=False)
    print("已保存 results/enrichment_summary.csv")

# ── Scaffold vs Random comparison ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("Split 对比（R²）:")
for algo in ["RF", "XGB", "SVR"]:
    for tgt in ["b2", "b1"]:
        sc_key   = f"{algo}_{tgt}_scaffold"
        rnd_key  = f"{algo}_{tgt}_random"
        ssc_key  = f"{algo}_{tgt}_stratified_scaffold"
        cv_key   = f"{algo}_{tgt}_cv"
        r2_sc    = r2_score(results[sc_key]["y_test"],  results[sc_key]["y_pred"])  if sc_key  in results else float("nan")
        r2_rnd   = r2_score(results[rnd_key]["y_test"], results[rnd_key]["y_pred"]) if rnd_key in results else float("nan")
        r2_ssc   = r2_score(results[ssc_key]["y_test"], results[ssc_key]["y_pred"]) if ssc_key in results else float("nan")
        r2_cv    = results[cv_key]["cv_r2_mean"] if cv_key in results else float("nan")
        print(f"  {algo} {tgt.upper()}: CV={r2_cv:.3f} | strat-scaffold={r2_ssc:.3f} | "
              f"scaffold={r2_sc:.3f} | random={r2_rnd:.3f}")
