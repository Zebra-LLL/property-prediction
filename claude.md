# CYP11B2 / CYP11B1 QSAR 建模项目

## 项目背景

针对醛固酮合酶（CYP11B2）抑制剂的选择性预测建模项目。
核心挑战是 CYP11B2 与 CYP11B1 之间 93% 的序列同源性，需要通过模型辅助筛选对 CYP11B2 有活性、对 CYP11B1 选择性强的化合物。

## 数据文件

- `data.in`：原始活性数据，制表符分隔，三列：
  - 第一列：SMILES
  - 第二列：CYP11B1 IC50（μM）
  - 第三列：CYP11B2 IC50（μM）
- 数据来源：专利公开数据，全部为正的精确数值（已排除截断值和 n.d.）
- 数值为 0 的条目含义不明确，需在数据清洗阶段丢弃

## 建模指南

详细步骤、代码和注意事项见 `CYP11B_QSAR_modeling_guide.md`，执行任何建模任务前必须先完整阅读该文件。

## 建模策略

- 建立两个**独立的回归模型**：Model_B2 预测 pIC50(CYP11B2)，Model_B1 预测 pIC50(CYP11B1)
- 不建选择性模型（训练数据缺乏选择性阴性样本）
- 联合使用两个模型：B2 预测值高 + B1 预测值低 = 优先化合物
- 活性单位统一换算为 pIC50，公式：`pIC50 = 6 - log10(IC50_uM)`

## 输出规范

- 清洗后数据：`data_cleaned.csv`
- 模型文件：`models/` 目录（.pkl 格式）
- 图表和评估结果：`results/` 目录
- 每个步骤生成独立的 Python 脚本文件，不使用 notebook

## 技术栈

```
rdkit, pandas, numpy, scikit-learn, xgboost, lightgbm, deepchem, shap, matplotlib, scipy
```

## 环境配置（conda）

> **推荐 Python 3.10**：对 RDKit、XGBoost、SHAP 兼容性最佳；DeepChem 需要单独处理（见下文）。

### 快速创建（使用 environment.yml）

```bash
conda env create -f environment.yml
conda activate cyp11b_qsar
```

### 手动创建

```bash
# 1. 创建环境（rdkit 必须走 conda-forge，pip 安装易出错）
conda create -n cyp11b_qsar python=3.10 -y
conda activate cyp11b_qsar

# 2. 核心依赖（conda-forge 渠道）
conda install -c conda-forge \
    rdkit=2024.03 \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    scipy \
    joblib \
    -y

# 3. 机器学习模型
pip install xgboost lightgbm

# 4. 可解释性
pip install shap

# 5. （可选）DeepChem / AttentiveFP
# DeepChem 依赖 PyTorch，版本敏感，单独安装以避免冲突
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install deepchem
```

### 各依赖说明

| 包 | 用途 | 安装渠道 | 备注 |
|----|------|---------|------|
| `rdkit` | SMILES解析、指纹、描述符、骨架 | **conda-forge** | 不能用 pip 安装，否则易报 segfault |
| `pandas` / `numpy` | 数据处理 | conda-forge | — |
| `scikit-learn` | RandomForest、数据划分、预处理 | conda-forge | — |
| `scipy` | Pearson r 等统计指标 | conda-forge | — |
| `joblib` | 模型序列化（.pkl） | conda-forge | scikit-learn 自带，无需单独安装 |
| `xgboost` | XGBoost 回归模型 | pip | — |
| `lightgbm` | 备选梯度提升模型（当前未用） | pip | — |
| `shap` | 模型可解释性（TreeExplainer） | pip | — |
| `matplotlib` | parity plot、SHAP 图 | conda-forge | — |
| `deepchem` | AttentiveFP 图神经网络（可选） | pip | 需先装 PyTorch；CPU版本即可 |

## 执行原则

- 遇到报错自行调试修复，无需中断询问
- **主要评估指标：Stratified-scaffold 5-fold CV R²**（每个骨架组内按比例分配到各折）
- 不使用固定测试集；最终模型在全量有效数据上训练
- 所有模型均需输出：R²（CV）以及 parity plot

---

## 已完成进度

### 已创建脚本（branch: `claude/read-claude-md-EiqFB`）

| 脚本 | 功能 | 输出 |
|------|------|------|
| `step1_preprocess.py` | 数据清洗、SMILES标准化、pIC50换算、InChIKey去重 | `data_cleaned.csv` |
| `step2_features.py` | ECFP4(2048-bit) + RDKit 2D描述符（方差过滤） | `features/X_combined.npy` (2189d) |
| `step2b_feature_select.py` | 三阶段特征筛选（方差/相关性/RF重要性）+ CV前后对比 | `features/X_selected.npy` (300d) |
| `step3_split.py` | Bemis-Murcko scaffold（保留文件供参考，step4不再使用测试集） | `splits/*.npz` |
| `step4_train.py` | RF + XGB + SVR，stratified-scaffold 5-fold CV + 最终全量训练 | `models/*_final.pkl`, `models/train_results.pkl` |
| `train_final_models.py` | 用最优Optuna参数在全量数据上训练最终SVR模型 | `models/final_model_B2.pkl`, `models/final_model_B1.pkl`, `models/feature_selector.pkl` |
| `step6_shap.py` | XGBoost SHAP全局重要性图 + 单化合物force plot | `results/shap_*.png`, `results/feature_importance_*.csv` |
| `plot_cv_parity.py` | Stratified-scaffold CV parity plot（XGB + SVR） | `figures/cv_parity_xgb_svr.png` |
| `plot_ef.py` | EF@10% / EF@20% 富集因子（stratified-scaffold CV） | `figures/ef_barplot.png` |
| `run_pipeline.sh` | 一键执行全流程 | — |

**执行方式**：将 `data.in` 放入根目录后运行 `bash run_pipeline.sh`

### 建模策略（2026-03-25 更新）

**评估方式：Stratified-scaffold 5-fold CV**
- 每个 Bemis-Murcko 骨架组内的化合物按比例（round-robin）分配到 5 折
- 每折都能见到所有主要骨架，结构多样性与全集一致
- 比纯随机 CV 更真实（消除同骨架化合物泄漏），比 GroupKFold 更温和（不做纯外推）

**不设固定测试集**，CV R² 为唯一报告指标；最终模型在全量有效数据上训练。

### 当前模型性能（Stratified-scaffold 5-fold CV，X_selected 300d）

**数据：** 457条原始数据 → 415个去重化合物；B2有效371个，B1有效334个；独特骨架119个

| 模型 | CYP11B2 CV R² | CYP11B1 CV R² |
|------|--------------|--------------|
| RF   | 0.422 ± 0.080 | 0.373 ± 0.059 |
| XGB  | 0.451 ± 0.059 | 0.419 ± 0.063 |
| SVR  | **0.575 ± 0.064** | **0.502 ± 0.063** |

**特征筛选前后对比（SVR）：**

| 特征集 | B2 CV R² | B1 CV R² |
|--------|---------|---------|
| X_combined (2189d) | 0.491 | 0.440 |
| X_selected (300d)  | **0.575** | **0.502** |
| Δ | +0.084 | +0.062 |

**富集因子（X_selected，stratified-scaffold CV）：**

| 模型 | 靶标 | 阈值 | EF@10% | EF@20% |
|------|------|------|--------|--------|
| XGB | CYP11B2 | pIC50 > 7.5 | 1.56 | 1.54 |
| SVR | CYP11B2 | pIC50 > 7.5 | 1.52 | 1.52 |
| XGB | CYP11B1 | pIC50 < 6.0 | 2.08 | 1.82 |
| SVR | CYP11B1 | pIC50 < 6.0 | 1.95 | 1.88 |

**运行环境：** 系统 Python 3.11，rdkit-2025.9.6、xgboost-3.2.0、shap-0.51.0、scikit-learn-1.8.0

### 特征集

三阶段筛选（`step2b_feature_select.py`）：

| 阶段 | 操作 | 剩余特征数 |
|------|------|-----------|
| 原始 | X_combined | 2189 |
| 方差过滤 | threshold=0.01 | 539 |
| 相关性过滤 | \|r\|>0.95 去重 | 466 |
| RF重要性 | 平均B2+B1，top 300 | **300** |

保存为 `features/X_selected.npy`（415 × 300）

### 最终模型文件（用于推断）

```
models/feature_selector.pkl    # IndexSelector：X_combined (2189d) → X_selected (300d)
models/final_model_B2.pkl      # Pipeline(StandardScaler + SVR)，CYP11B2，训练于371个样本
models/final_model_B1.pkl      # Pipeline(StandardScaler + SVR)，CYP11B1，训练于334个样本
```

**SVR 最优超参数（Optuna stratified-scaffold CV 搜索）：**

| 模型 | C | ε | γ | 训练集样本数 |
|------|---|---|---|------------|
| B2   | 3.41 | 0.112 | 0.00144 | 371 |
| B1   | 639  | 0.28  | 0.00124 | 334 |

**使用方式：**
```python
import joblib, numpy as np

selector = joblib.load("models/feature_selector.pkl")
model_b2 = joblib.load("models/final_model_B2.pkl")
model_b1 = joblib.load("models/final_model_B1.pkl")

# X_new_combined: (n_compounds, 2189) — 用 step2_features.py 计算
X_new_sel = selector.transform(X_new_combined)   # (n_compounds, 300)
pred_b2   = model_b2.predict(X_new_sel)          # pIC50 CYP11B2
pred_b1   = model_b1.predict(X_new_sel)          # pIC50 CYP11B1
```

**CV 评估模型（供参考，不用于推断）：**
```
models/rf_{b2,b1}_final.pkl    # Random Forest
models/xgb_{b2,b1}_final.pkl   # XGBoost
models/svr_{b2,b1}_final.pkl   # SVR（同参数，与 final_model 等价）
```

---

## 已知问题与注意事项

### SHAP + XGBoost 版本兼容性问题（已修复）

**现象：** 本地 conda 环境运行 `step6_shap.py` 报错：
```
ValueError: could not convert string to float: '[5E-1]'
```

**根因：** XGBoost ≥ 2.0 将 `base_score` 以 `[5E-1]` 括号格式存入模型 JSON，SHAP < 0.44 无法解析该格式。脚本中原有的 `patch_xgb_base_score` 函数调用了 `booster.load_model(bytearray(...))` 试图修复，但 XGBoost 将 bytearray 输入视为二进制 UBJ 格式而非 JSON，导致修复静默失败（被 `except` 吞掉）。

**修复（2026-03-25）：**
1. `step6_shap.py`：修正 `patch_xgb_base_score`，改用临时文件写入 + `load_model(tmp_path)` 触发 XGBoost 的 JSON 加载路径
2. `environment.yml`：钉住 `shap>=0.44.0`（该版本原生支持 `[5E-1]` 格式，无需 workaround）和 `xgboost>=2.0`

### 单位假设硬编码（重要）

`step1_preprocess.py` 中的 pIC50 换算公式固定假设 **IC50 单位为 μM**：

```python
pIC50 = 6 - log10(IC50_uM)   # 仅适用于 μM 单位
```

| 实际单位 | 套用现有公式的结果 |
|---------|-----------------|
| μM | 正确 |
| nM | pIC50 **偏低 3 个单位**（活性被严重低估） |
| mM | pIC50 **偏高 3 个单位** |

**当前数据（专利来源）明确标注单位为 μM，公式正确。**
若后续引入其他来源数据（如 ChEMBL、内部实验），必须先统一单位再运行脚本，或在 `step1_preprocess.py` 中添加单位转换逻辑。

### Morgan 指纹 API 迁移（RDKit 2022+）

`AllChem.GetMorganFingerprintAsBitVect()` 在 RDKit 新版本中已废弃，替换为 `rdFingerprintGenerator.GetMorganGenerator`。

已修改 `step2_features.py`：

```python
# 修改前（旧写法）
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

def mol_to_ecfp4(smi, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# 修改后（新写法）
from rdkit.Chem import Descriptors, rdFingerprintGenerator

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)  # 模块级，只创建一次

def mol_to_ecfp4(smi):
    mol = Chem.MolFromSmiles(smi)
    return _morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)  # 直接返回 numpy array
```

改动要点：
- `GetMorganGenerator` + `GetFingerprintAsNumPy` 直接返回 numpy 数组，无需 `DataStructs.ConvertToNumpyArray`
- generator 在模块级创建一次，避免每个分子重复初始化
- 移除 `AllChem` 和 `DataStructs` 的导入

### XGBoost base_score 格式问题（XGBoost ≥ 2.0 + SHAP 兼容性）

XGBoost ≥ 2.0 将 `base_score` 存储为带括号的字符串（如 `'[7.754658E0]'`），SHAP 的 `TreeExplainer` 对其执行 `float(...)` 时抛出：
```
ValueError: could not convert string to float: '[7.754658E0]'
```

已修改两处：
- **`step4_train.py`**：在 `XGBRegressor()` 构造函数中显式传入 `base_score=0.5`，避免未来保存的模型触发此问题
- **`step6_shap.py`**：新增 `patch_xgb_base_score()` 函数，加载模型后立即修复 booster config 中的括号格式（兼容已有的保存模型）

```python
# step6_shap.py 中的修复函数（必须用 save_raw/load_model，不能用 save_config/load_config）
def patch_xgb_base_score(model):
    booster = model.get_booster()
    raw = json.loads(booster.save_raw("json").decode())   # 读模型二进制数据
    lmp = raw["learner"]["learner_model_param"]
    bs = lmp.get("base_score", "0.5")
    if isinstance(bs, str) and "[" in bs:                 # '[5E-1]' -> '0.5'
        lmp["base_score"] = str(float(bs.strip("[]")))
        booster.load_model(bytearray(json.dumps(raw).encode()))
    return model
```

> **注意**：`save_config()/load_config()` 操作的是训练超参数，不含 `learner_model_param`；
> SHAP 通过 `save_raw('json')` 读取模型数据，因此 patch 必须走 `save_raw/load_model` 路径。

### XGBoost early_stopping_rounds 参数位置修正（XGBoost ≥ 2.0）

XGBoost ≥ 2.0 中 `early_stopping_rounds` 必须在**构造函数**中传入，不能再传给 `fit()`，否则报：
```
TypeError: XGBModel.fit() got an unexpected keyword argument 'early_stopping_rounds'
```

已修改 `step4_train.py`：
```python
# 修改前（XGBoost < 2.0 写法，现已报错）
xgb_model = xgb.XGBRegressor(n_estimators=500, ...)
xgb_model.fit(X_tr, y_tr, eval_set=[...], early_stopping_rounds=50)

# 修改后（XGBoost ≥ 2.0 正确写法）
xgb_model = xgb.XGBRegressor(n_estimators=500, ..., early_stopping_rounds=50)
xgb_model.fit(X_tr, y_tr, eval_set=[...])
```

### SMILES 标准化函数修正（RDKit 2024 兼容性）

`rdMolStandardize.Standardizer()` 在 RDKit 2024 中对部分分子抛出异常，被 `except Exception: return None` 静默吞掉，导致全部 457 个 SMILES 返回 None，所有化合物被丢弃。

已修改 `step1_preprocess.py`：

- **移除** `SaltRemover.SaltRemover()` + `Standardizer()`（问题根源）
- **替换**为 `LargestFragmentChooser` + `Normalizer` + `Uncharger` 的显式调用链
- **对象实例化移至函数外**，避免每次调用重新加载盐表（原代码每个分子都创建一次，既慢又易出错）
- 每步操作后添加 `None` 检查

```python
# 修改前（问题代码）
def standardize_smiles(smi):
    remover = SaltRemover.SaltRemover()        # 每次调用重新加载盐表
    mol = remover.StripMol(mol)
    standardizer = rdMolStandardize.Standardizer()  # RDKit 2024 下易抛异常
    mol = standardizer.standardize(mol)
    ...

# 修改后（正确）
_normalizer = rdMolStandardize.Normalizer()    # 模块级实例化，只执行一次
_uncharger  = rdMolStandardize.Uncharger()
_chooser    = rdMolStandardize.LargestFragmentChooser()

def standardize_smiles(smi):
    mol = _chooser.choose(mol)      # 取最大片段（desalt）
    mol = _normalizer.normalize(mol)
    mol = _uncharger.uncharge(mol)
    ...
```

### B1/B2 列顺序修正

`data.in` 实际列顺序为：`smiles | CYP11B1 | CYP11B2`（B1 在前，B2 在后），与建模指南示例相反。

已修改 `step1_preprocess.py:20`：
```python
# 修改前（错误）
names=["smiles", "IC50_B2_uM", "IC50_B1_uM"]
# 修改后（正确）
names=["smiles", "IC50_B1_uM", "IC50_B2_uM"]
```

### 数据文件名修正

原始数据文件实际名称为 `data.in`，而非建模指南中示例的 `data.txt`，分隔符不变（制表符）。

已修改的文件：
- `step1_preprocess.py`：`read_csv("data.txt")` → `read_csv("data.in")`
- `run_pipeline.sh`：文件存在检查及提示信息均已更新
