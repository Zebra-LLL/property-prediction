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
- 优先使用 scaffold split 评估模型，随机划分仅作对比参考
- 所有模型均需输出：R²、RMSE、MAE、Pearson r 以及 parity plot

---

## 已完成进度

### 已创建脚本（branch: `claude/read-claude-md-EiqFB`）

| 脚本 | 功能 | 输出 |
|------|------|------|
| `step1_preprocess.py` | 数据清洗、SMILES标准化、pIC50换算、InChIKey去重 | `data_cleaned.csv` |
| `step2_features.py` | ECFP4(2048-bit) + RDKit 2D描述符（方差过滤） | `features/*.npy`, `features/*.pkl` |
| `step3_split.py` | Bemis-Murcko scaffold split + 随机划分 | `splits/*.npz` |
| `step4_train.py` | RF + XGBoost，B2/B1 各两套（scaffold + random），共8个模型 | `models/*.pkl` |
| `step5_evaluate.py` | R²、RMSE、MAE、Pearson r，parity plot，富集因子 | `results/metrics_summary.csv`, `results/parity_*.png` |
| `step6_shap.py` | XGBoost SHAP全局重要性图 + 单化合物force plot | `results/shap_*.png`, `results/feature_importance_*.csv` |
| `run_pipeline.sh` | 一键执行全流程 | — |

**执行方式**：将 `data.in` 放入根目录后运行 `bash run_pipeline.sh`

### 待完成

- [ ] 提供 `data.in` 后实际运行流程并检查输出
- [ ] 根据运行结果决定是否引入 RDKit 描述符或组合特征
- [ ] 如需可解释性分析，切换为描述符特征后重新运行 SHAP

---

## 已知问题与注意事项

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
