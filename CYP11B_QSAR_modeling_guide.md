# CYP11B2 / CYP11B1 QSAR 建模指南

## 数据概况

| 项目 | 详情 |
|------|------|
| 数据来源 | 专利公开数据 |
| 总化合物数 | 407个（重新编号） |
| 数据格式 | SMILES + pIC50_B2 + pIC50_B1 三列文本文件 |
| B2可用数据 | 406个精确值 |
| B1可用数据 | 359个精确值（另有8个截断值丢弃） |
| 单位 | 统一换算为 pIC50（−log10[IC50/M]） |

建模策略：两个独立回归模型分别预测 pIC50(CYP11B2) 和 pIC50(CYP11B1)，联合使用时以 B2 预测值高、B1 预测值低作为筛选标准。

---

## 环境依赖

```bash
pip install rdkit pandas numpy scikit-learn xgboost lightgbm \
            deepchem torch dgl dgllife \
            shap matplotlib seaborn scipy
```

---

## 第一步：数据读取与预处理

### 1.1 读取原始数据

```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

df = pd.read_csv("data.txt", sep="\t", header=0,
                 names=["smiles", "IC50_B2_uM", "IC50_B1_uM"])
print(f"原始行数: {len(df)}")
```

### 1.2 活性数据处理

截断数据（含 `>` 符号）、`n.d.` 以及数值为 0 的条目一律丢弃，仅保留正的精确数值。
值为 0 的条目含义不明确（可能是未测试、仪器下限或录入错误），不可直接用于 pIC50 换算（log(0) 无定义），保守处理为 NaN。

```python
def parse_activity(val):
    """
    返回 float 或 NaN。
    以下情况均返回 NaN：
      - 缺失值（空白、NaN）
      - 截断值（> 或 <）
      - n.d.（未测定）
      - 数值为 0（含义不明，保守丢弃）
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.startswith(">") or s.startswith("<") or s.lower() == "n.d.":
        return np.nan
    try:
        v = float(s)
        return np.nan if v == 0 else v   # 0值丢弃
    except ValueError:
        return np.nan

df["IC50_B2_uM"] = df["IC50_B2_uM"].apply(parse_activity)
df["IC50_B1_uM"] = df["IC50_B1_uM"].apply(parse_activity)

# 统计被丢弃的0值数量，便于核查
zero_b2 = (df["IC50_B2_uM"] == 0).sum()  # parse后已为NaN，此行仅作提示
print("注：原始数据中值为0的条目已丢弃，请手动确认数量是否符合预期")

# 换算为 pIC50
# IC50单位为μM，换算公式：pIC50 = -log10(IC50 * 1e-6) = 6 - log10(IC50_uM)
df["pIC50_B2"] = df["IC50_B2_uM"].apply(
    lambda x: 6 - np.log10(x) if pd.notna(x) and x > 0 else np.nan
)
df["pIC50_B1"] = df["IC50_B1_uM"].apply(
    lambda x: 6 - np.log10(x) if pd.notna(x) and x > 0 else np.nan
)

print(f"B2有效数据: {df['pIC50_B2'].notna().sum()}")
print(f"B1有效数据: {df['pIC50_B1'].notna().sum()}")
```

### 1.3 SMILES 标准化与清洗

```python
def standardize_smiles(smi):
    """标准化SMILES：脱盐、去电荷、规范化"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        # 脱盐
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)

        # 标准化（去电荷、规范互变异构体）
        standardizer = rdMolStandardize.Standardizer()
        mol = standardizer.standardize(mol)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # 选最大片段
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)

        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None

df["smiles_std"] = df["smiles"].apply(standardize_smiles)
invalid = df["smiles_std"].isna().sum()
print(f"无效SMILES: {invalid} 个，已丢弃")
df = df.dropna(subset=["smiles_std"]).reset_index(drop=True)
```

### 1.4 去重处理

结构完全相同的化合物取 pIC50 几何均值（即 pIC50 算术均值）。

```python
from rdkit.Chem import InchiInfo
from rdkit.Chem.inchi import MolToInchiKey

def get_inchikey(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return MolToInchiKey(mol)

df["inchikey"] = df["smiles_std"].apply(get_inchikey)

# 对重复结构取均值
df_dedup = df.groupby("inchikey", as_index=False).agg(
    smiles_std=("smiles_std", "first"),
    pIC50_B2=("pIC50_B2", "mean"),
    pIC50_B1=("pIC50_B1", "mean"),
)
print(f"去重后化合物数: {len(df_dedup)}")
df_dedup.to_csv("data_cleaned.csv", index=False)
```

---

## 第二步：特征计算

同时计算两类特征，后续通过交叉验证决定使用哪种或组合使用。

### 2.1 Morgan Fingerprint（ECFP4）

```python
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

def mol_to_ecfp4(smi, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X_ecfp4 = np.vstack(df_dedup["smiles_std"].apply(mol_to_ecfp4).values)
print(f"ECFP4 特征矩阵: {X_ecfp4.shape}")
```

### 2.2 RDKit 2D 描述符

```python
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

desc_names = [name for name, _ in Descriptors.descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def mol_to_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(desc_names)
    return list(calculator.CalcDescriptors(mol))

X_desc_raw = np.array(
    df_dedup["smiles_std"].apply(mol_to_descriptors).tolist(),
    dtype=float
)

# 清理描述符：去除全NaN列、方差为0的列、填充剩余NaN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

imputer = SimpleImputer(strategy="median")
X_desc_imputed = imputer.fit_transform(X_desc_raw)

selector = VarianceThreshold(threshold=0.01)
X_desc = selector.fit_transform(X_desc_imputed)
print(f"RDKit描述符（过滤后）: {X_desc.shape}")
```

### 2.3 特征组合

```python
# 将ECFP4与描述符拼接作为第三套特征（可选）
X_combined = np.hstack([X_ecfp4, X_desc])
print(f"组合特征: {X_combined.shape}")
```

---

## 第三步：数据集划分

### 3.1 Scaffold Split（主要划分方式）

按 Bemis-Murcko 骨架划分，确保测试集包含模型未见过的骨架，评估真实泛化能力。

```python
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

df_dedup["scaffold"] = df_dedup["smiles_std"].apply(get_scaffold)

# 按骨架分组
scaffold_to_indices = defaultdict(list)
for i, scaffold in enumerate(df_dedup["scaffold"]):
    scaffold_to_indices[scaffold].append(i)

# 按骨架大小排序（大骨架优先分配到训练集）
scaffolds_sorted = sorted(scaffold_to_indices.items(),
                          key=lambda x: len(x[1]), reverse=True)

train_idx, val_idx, test_idx = [], [], []
total = len(df_dedup)

for scaffold, indices in scaffolds_sorted:
    if len(test_idx) / total < 0.1:
        test_idx.extend(indices)
    elif len(val_idx) / total < 0.1:
        val_idx.extend(indices)
    else:
        train_idx.extend(indices)

print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")
```

### 3.2 随机划分（用于对比基线）

```python
from sklearn.model_selection import train_test_split

idx_all = np.arange(len(df_dedup))
idx_train_r, idx_test_r = train_test_split(idx_all, test_size=0.2,
                                            random_state=42)
idx_train_r, idx_val_r = train_test_split(idx_train_r, test_size=0.125,
                                           random_state=42)
# 比例约为 train:val:test = 7:1:2
```

> **注意**：随机划分的评估指标通常会显著高于 scaffold split。两者差距越大，说明模型的泛化能力越依赖于记忆已知骨架，而非真正学到了构效关系。以 scaffold split 的结果为准。

---

## 第四步：建模

以下以 B2 模型为例，B1 模型流程完全相同，替换目标变量即可。

### 4.1 准备训练数据

```python
# 以 scaffold split + ECFP4 为例
target = "pIC50_B2"  # 切换为 "pIC50_B1" 即为B1模型

mask_b2 = df_dedup[target].notna()
X = X_ecfp4[mask_b2]
y = df_dedup.loc[mask_b2, target].values

valid_train = [i for i in train_idx if mask_b2.iloc[i]]
valid_val   = [i for i in val_idx   if mask_b2.iloc[i]]
valid_test  = [i for i in test_idx  if mask_b2.iloc[i]]

# 重新映射索引（mask_b2过滤后行号变化）
mask_positions = np.where(mask_b2)[0]
pos_to_new = {old: new for new, old in enumerate(mask_positions)}

X_train = X[[pos_to_new[i] for i in valid_train if i in pos_to_new]]
y_train = y[[pos_to_new[i] for i in valid_train if i in pos_to_new]]
X_val   = X[[pos_to_new[i] for i in valid_val   if i in pos_to_new]]
y_val   = y[[pos_to_new[i] for i in valid_val   if i in pos_to_new]]
X_test  = X[[pos_to_new[i] for i in valid_test  if i in pos_to_new]]
y_test  = y[[pos_to_new[i] for i in valid_test  if i in pos_to_new]]
```

### 4.2 Random Forest（首选 baseline）

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

rf = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_val  = rf.predict(X_val)
y_pred_test = rf.predict(X_test)

print(f"[RF] Val  R²={r2_score(y_val, y_pred_val):.3f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_val, y_pred_val)):.3f}")
print(f"[RF] Test R²={r2_score(y_test, y_pred_test):.3f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
```

### 4.3 XGBoost

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

y_pred_test_xgb = xgb_model.predict(X_test)
print(f"[XGB] Test R²={r2_score(y_test, y_pred_test_xgb):.3f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred_test_xgb)):.3f}")
```

### 4.4 超参数调优（以 RF 为例）

```python
from sklearn.model_selection import cross_val_score, KFold

param_grid = {
    "n_estimators": [200, 500],
    "max_features": ["sqrt", 0.3],
    "min_samples_leaf": [1, 2, 5],
    "max_depth": [None, 20],
}

from sklearn.model_selection import GridSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=42),
    param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"最优参数: {grid_search.best_params_}")
print(f"CV R²: {grid_search.best_score_:.3f}")
```

### 4.5 AttentiveFP（可选，GNN方法）

DeepChem 封装版，适合小数据。

```python
import deepchem as dc
from deepchem.models import AttentiveFPModel

# 构建 DeepChem 数据集
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
smiles_list = df_dedup["smiles_std"].values
y_b2 = df_dedup["pIC50_B2"].values

X_dc = featurizer.featurize(smiles_list)
dataset = dc.data.NumpyDataset(X=X_dc, y=y_b2.reshape(-1, 1))

# 使用 scaffold splitter
splitter = dc.splits.ScaffoldSplitter()
train_dc, val_dc, test_dc = splitter.train_val_test_split(
    dataset, frac_train=0.7, frac_valid=0.1, frac_test=0.2
)

model_afp = AttentiveFPModel(
    n_tasks=1,
    mode="regression",
    num_layers=2,
    num_timesteps=2,
    graph_feat_size=200,
    dropout=0.2,
    learning_rate=1e-3,
    batch_size=32,
    model_dir="./afp_b2"
)

model_afp.fit(train_dc, nb_epoch=100)

pred_val = model_afp.predict(val_dc)
r2_afp = r2_score(val_dc.y.flatten(), pred_val.flatten())
print(f"[AttentiveFP] Val R²={r2_afp:.3f}")
```

---

## 第五步：模型评估

### 5.1 评估指标

```python
from scipy import stats

def evaluate_model(y_true, y_pred, split_name="Test"):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = np.mean(np.abs(y_true - y_pred))
    r, _ = stats.pearsonr(y_true, y_pred)
    print(f"[{split_name}] R²={r2:.3f} | RMSE={rmse:.3f} | "
          f"MAE={mae:.3f} | Pearson r={r:.3f}")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": r}
```

### 5.2 预测值 vs 实测值散点图

```python
import matplotlib.pyplot as plt

def plot_parity(y_true, y_pred, title="Parity Plot", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue")

    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.plot(lims, [l + 0.5 for l in lims], "r--", linewidth=0.8,
            alpha=0.5, label="±0.5 pIC50")
    ax.plot(lims, [l - 0.5 for l in lims], "r--", linewidth=0.8, alpha=0.5)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Experimental pIC50"); ax.set_ylabel("Predicted pIC50")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

plot_parity(y_test, y_pred_test, title="RF - CYP11B2 (Scaffold Split)",
            save_path="parity_rf_b2.png")
```

### 5.3 富集分析（Enrichment Factor）

评估模型在筛选高活性化合物时的实用价值。

```python
def enrichment_factor(y_true, y_pred, top_frac=0.1, activity_cutoff=7.5):
    """
    EF = (命中率_top_n / 随机命中率)
    命中定义：实测 pIC50 > activity_cutoff
    """
    n_total = len(y_true)
    n_top = max(1, int(n_total * top_frac))

    # 按预测值排序，取top N
    top_idx = np.argsort(y_pred)[::-1][:n_top]

    hits_in_top = np.sum(y_true[top_idx] > activity_cutoff)
    hits_total  = np.sum(y_true > activity_cutoff)

    random_rate = hits_total / n_total
    top_rate    = hits_in_top / n_top

    ef = top_rate / random_rate if random_rate > 0 else 0
    print(f"EF@{int(top_frac*100)}%: {ef:.2f}x  "
          f"(top {n_top}中命中{hits_in_top}/{n_top}, "
          f"总命中率{random_rate*100:.1f}%)")
    return ef

ef = enrichment_factor(y_test, y_pred_test, top_frac=0.1, activity_cutoff=7.5)
```

### 5.4 模型对比汇总

```python
results = {}
for name, (y_pred, y_true) in {
    "RF_scaffold":  (y_pred_test,     y_test),
    "XGB_scaffold": (y_pred_test_xgb, y_test),
    # 随机划分结果另行计算后填入
}.items():
    results[name] = evaluate_model(y_true, y_pred, split_name=name)

results_df = pd.DataFrame(results).T
print(results_df.round(3))
```

---

## 第六步：可解释性分析

### 6.1 SHAP 值分析（RF / XGBoost）

```python
import shap

# XGBoost SHAP（速度快）
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 全局特征重要性
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.savefig("shap_summary_b2.png", dpi=150, bbox_inches="tight")
plt.show()

# 单个化合物解释（对预测结果最高的化合物）
top_compound_idx = np.argmax(y_pred_test_xgb)
shap.force_plot(
    explainer.expected_value,
    shap_values[top_compound_idx],
    X_test[top_compound_idx],
    matplotlib=True
)
```

### 6.2 特征重要性（RF）

```python
importances = rf.feature_importances_
# 仅对描述符模型有意义（ECFP4 位不可直接解读）
# 如使用描述符特征，可提取 top 20 重要描述符
top_features_idx = np.argsort(importances)[::-1][:20]
```

---

## 第七步：模型保存与预测流程

### 7.1 保存模型

```python
import joblib

joblib.dump(rf,        "model_rf_b2.pkl")
joblib.dump(xgb_model, "model_xgb_b2.pkl")
joblib.dump(imputer,   "desc_imputer.pkl")
joblib.dump(selector,  "desc_selector.pkl")
```

### 7.2 新化合物预测函数

```python
def predict_new_compounds(smiles_list, model_b2, model_b1,
                          featurizer="ecfp4", **kwargs):
    """
    输入 SMILES 列表，返回 B2 和 B1 的 pIC50 预测值及推断选择性。
    """
    results = []
    for smi in smiles_list:
        # 标准化
        smi_std = standardize_smiles(smi)
        if smi_std is None:
            results.append({"smiles": smi, "pIC50_B2": np.nan,
                             "pIC50_B1": np.nan, "delta_pIC50": np.nan})
            continue

        # 特征
        feat = mol_to_ecfp4(smi_std).reshape(1, -1)

        pred_b2 = float(model_b2.predict(feat)[0])
        pred_b1 = float(model_b1.predict(feat)[0])
        delta   = pred_b2 - pred_b1  # 正值 = 对B2选择性更强

        results.append({
            "smiles":      smi,
            "pIC50_B2":    round(pred_b2, 2),
            "pIC50_B1":    round(pred_b1, 2),
            "delta_pIC50": round(delta, 2),
            "selectivity_fold": round(10 ** delta, 1)
        })

    return pd.DataFrame(results).sort_values("pIC50_B2", ascending=False)
```

---

## 关键判断节点与预期指标

### 各模型预期性能范围

| 划分方式 | 预期 R²(B2) | 预期 RMSE(B2) | 判断 |
|----------|-------------|----------------|------|
| 随机划分 | 0.65 ~ 0.80 | 0.30 ~ 0.45 | baseline参考 |
| Scaffold split | 0.40 ~ 0.65 | 0.45 ~ 0.65 | 真实泛化能力 |

> Scaffold split 的 R² < 0.4 说明模型泛化能力不足，需要检查特征或增加数据多样性。Scaffold split R² > 0.6 则是非常好的结果。

### 特征选择建议

| 情况 | 建议特征 |
|------|---------|
| 同系列化合物（R基优化） | ECFP4（对局部结构变化敏感） |
| 多骨架数据集 | RDKit 描述符 或 ECFP4 + 描述符组合 |
| 追求可解释性 | RDKit 描述符 + SHAP |
| 追求最佳预测性能 | 以上三套特征分别建模，取CV最优 |

### 常见问题与排查

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| 随机/scaffold 差距 > 0.3 | 过拟合，训练集骨架记忆 | 减少树深度，增大 min_samples_leaf |
| 测试集 RMSE > 1.0 | 测试集包含训练集未覆盖的活性区间 | 检查活性分布，考虑分层划分 |
| B1 模型显著差于 B2 | B1 数据量少（359 vs 406），且 pIC50 分布更宽 | 正常现象，可适当降低 B1 评估期望 |
| SHAP 显示某个 bit 贡献最大 | ECFP4 bit 不可解读 | 切换为描述符特征做解释性分析 |

---

## 推荐执行顺序

```
1. 数据清洗与标准化          → 生成 data_cleaned.csv
2. 计算 ECFP4 特征            → 作为首选特征
3. Scaffold split 划分        → 作为主要评估
4. RF 建模（B2 和 B1 各一个）  → 作为 baseline
5. XGBoost 建模               → 与 RF 对比
6. 选最优模型做 SHAP 分析      → 理解构效关系
7. 随机划分重复步骤 3-5       → 与 scaffold split 对比，估计泛化损失
8. 包装预测函数               → 对接虚筛流程
```

---

## 注意事项

- 所有模型均在**专利公开数据**上训练，活性分布偏向已优化化合物，对全新骨架的预测需谨慎解读
- 选择性（delta_pIC50）为两个独立模型预测值之差，误差会叠加，置信度低于单靶标预测
- 建议对预测结果设置置信区间（RF 的 `predict` 可通过各棵树的标准差估计不确定度）
- 如果后续获得内部实验数据，应将其加入训练集并重新评估模型，专利数据与内部数据的测定条件可能存在差异
